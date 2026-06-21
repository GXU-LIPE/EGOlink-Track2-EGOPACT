#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Write V9.5 comparison/readiness reports for Track2."""
from __future__ import annotations

import json
import re
import time
from collections import Counter
from pathlib import Path

CODEX = Path("/home/data-gxu/acm/egolink2026-main/code/track2/codex")
RUN_ID = "V9_5_memory_deepseek_rerank_validation_A_medium_20260618_1218_noproxy"
VERSION = "V9_5_memory_deepseek_rerank"
RUN_DIR = CODEX / "runs" / VERSION / RUN_ID
V945 = "V9_4_5_visual_retrieval_fix_validation_A_medium_20260618_1014"


def load_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def pct(x):
    return f"{float(x) * 100:.2f}%"


def iter_jsonl(root: Path):
    if not root.exists():
        return
    for path in sorted(root.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def tool_log_counts(run_dir: Path):
    rows = []
    for path in sorted((run_dir / "logs").glob("*.log")):
        text = path.read_text(encoding="utf-8", errors="replace")
        calls = len(re.findall(r"\[Tool Execution\] Calling:", text))
        stats = [int(x) for x in re.findall(r"Task\s+\d+:.*?(\d+)\s+tool calls", text)]
        rows.append({"log": path.name, "calls": calls, "max_task_calls": max(stats or [0]), "bytes": path.stat().st_size})
    return rows


def main() -> None:
    ts = time.strftime("%Y%m%d_%H%M%S")
    eval_v95 = load_json(RUN_DIR / "eval_summary.json", {"summary": {}, "rows": []})
    eval_v945 = load_json(CODEX / "runs" / "V9_4_5_visual_retrieval_fix" / V945 / "eval_summary.json", {"summary": {}, "rows": []})
    s95 = eval_v95.get("summary", {})
    s945 = eval_v945.get("summary", {})

    trim_events = [e for e in iter_jsonl(RUN_DIR / "v8_events") if e.get("reason") == "retail_broad_scan_candidate_trimmed"]
    multi_count = sum(1 for e in iter_jsonl(RUN_DIR / "v8_events") if e.get("reason") == "multicandidate_score")
    trim_original = sum(int(e.get("original_calls") or 0) for e in trim_events)
    trim_repaired = sum(int(e.get("repaired_calls") or 0) for e in trim_events)
    timeout_count = 0
    soft_fail_count = 0
    for path in (RUN_DIR / "logs").glob("*.log"):
        text = path.read_text(encoding="utf-8", errors="replace")
        timeout_count += text.count("ReadTimeout")
        soft_fail_count += text.count("Direct API Soft Failure")

    logs95 = tool_log_counts(RUN_DIR)
    logs945 = tool_log_counts(CODEX / "runs" / "V9_4_5_visual_retrieval_fix" / V945)
    log95_by = {x["log"]: x for x in logs95}
    log945_by = {x["log"]: x for x in logs945}

    report_lines = [
        f"# V9_5 DeepSeek Reranker A_medium {ts}",
        "",
        f"- run_id: `{RUN_ID}`",
        f"- version: `{VERSION}`",
        "- final_submission: not submitted",
        "- protected_best_updated: no",
        "- DeepSeek online crosscheck: not used; remote `.deepseek_env` was absent, so no DeepSeek key was available.",
        "- GPT endpoint mode: no-proxy launcher, because local proxy `127.0.0.1:17897` timed out CONNECT to ai-pixel domains.",
        "",
        "## Metrics",
        "",
        "| version | valid | joint | result | tool | micro | correct/gt | interactions |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| V9_4 | 41 | 4.88% | 9.76% | 4.88% | 17.95% | - | - |",
        f"| V9_4_5 | {s945.get('valid', 0)} | {pct(s945.get('joint', 0))} | {pct(s945.get('result', 0))} | {pct(s945.get('tool', 0))} | {pct(s945.get('micro', 0))} | {s945.get('correct_calls', 0)}/{s945.get('gt_calls', 0)} | {s945.get('interaction_calls', 0)} |",
        f"| V9_5 | {s95.get('valid', 0)} | {pct(s95.get('joint', 0))} | {pct(s95.get('result', 0))} | {pct(s95.get('tool', 0))} | {pct(s95.get('micro', 0))} | {s95.get('correct_calls', 0)}/{s95.get('gt_calls', 0)} | {s95.get('interaction_calls', 0)} |",
        "",
        "## Telemetry",
        "",
        f"- multicandidate_score events: {multi_count}",
        f"- retail broad scan trim events: {len(trim_events)}",
        f"- trimmed tool calls: original={trim_original}, repaired={trim_repaired}",
        f"- ReadTimeout in completed no-proxy run: {timeout_count}",
        f"- Direct API Soft Failure in completed no-proxy run: {soft_fail_count}",
        "",
        "## Outcome",
        "",
        "- V9_5 successfully exercised the V9.5 candidate/rerank path and reduced several retail broad scans.",
        "- However, A_medium joint dropped versus V9_4_5 and the split was not identical: V9_5 skipped `kitchen2::33`, so valid=40.",
        "- Result success was roughly flat versus V9_4_5, micro was slightly higher, but joint/tool success regressed.",
        "- Do not run validation_B_holdout or final from V9_5.",
        "- Do not update protected best.",
        "",
        "## Hotspot Comparison",
        "",
        "| log | V9_4_5 calls | V9_5 calls | V9_4_5 bytes | V9_5 bytes |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in sorted(set(log945_by) | set(log95_by)):
        a = log945_by.get(name, {})
        b = log95_by.get(name, {})
        report_lines.append(f"| {name} | {a.get('calls', 0)} | {b.get('calls', 0)} | {a.get('bytes', 0)} | {b.get('bytes', 0)} |")

    report_lines += [
        "",
        "## Next Patch Direction",
        "",
        "- Freeze/materialize validation splits instead of recalculating live scenario JSON lengths; this prevents `kitchen2::33` drift.",
        "- Make candidate builder process-aware: trimming must preserve expected GT process families and not only reduce calls.",
        "- Add order-specific process candidates, because V9_5 did not repair order trajectory shape.",
        "- Add DeepSeek crosscheck key/env only if needed; current run proves the framework works without online DeepSeek.",
    ]

    report = CODEX / "reports" / f"V9_5_DEEPSEEK_RERANKER_{ts}.md"
    report.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    rerun = CODEX / "reports" / f"V9_VALIDATION_A_MEDIUM_RERUN_{ts}.md"
    rerun.write_text(
        "\n".join([
            f"# V9 Validation A Medium Rerun {ts}",
            "",
            "## Decision",
            "",
            "- Best candidate from this cycle: `V9_4_5_visual_retrieval_fix` for A_medium joint.",
            "- V9_5 is retained as an experimental patch but not promoted.",
            "- validation_B_holdout was not run because V9_5 did not beat V9_4_5 or protected V6 on A_medium.",
            "- final was not run.",
            "",
            "## Scores",
            "",
            f"- V9_4 medium corrected: joint 4.88%, result 9.76%, tool 4.88%, micro 17.95%",
            f"- V9_4_5 medium: joint {pct(s945.get('joint',0))}, result {pct(s945.get('result',0))}, tool {pct(s945.get('tool',0))}, micro {pct(s945.get('micro',0))}",
            f"- V9_5 medium no-proxy: joint {pct(s95.get('joint',0))}, result {pct(s95.get('result',0))}, tool {pct(s95.get('tool',0))}, micro {pct(s95.get('micro',0))}",
            "",
            "## Reports",
            "",
            f"- V9_4_5 report: `reports/V9_4_5_VISUAL_RETRIEVAL_FIX_20260618_115600.md`",
            f"- V9_5 report: `{report}`",
            f"- Raw evaluator report: `reports/V8_VALIDATION_A_MEDIUM_{RUN_ID}.md`",
        ]) + "\n",
        encoding="utf-8",
    )

    readiness = CODEX / "reports" / f"V9_NEXT_TOP1_READINESS_{ts}.md"
    readiness.write_text(
        "\n".join([
            f"# V9 Next Top1 Readiness {ts}",
            "",
            "## Status",
            "",
            "- Not top1-ready.",
            "- Protected best remains `V6_1_3_gpt55_guarded_endpoint`.",
            "- Best A_medium V9 candidate is `V9_4_5_visual_retrieval_fix`, but it is still far below top1-level reliability.",
            "- V9_5 reduced some retail scans but regressed joint success and had split drift.",
            "",
            "## Required Before Any Final",
            "",
            "- Fix validation split materialization.",
            "- Recover order1 process coverage above current 3-4/24 call match.",
            "- Preserve V9_4_5 joint while reducing retail broad scans.",
            "- Run A_medium and then B_holdout; only promote if both beat V6/protected best.",
            "- Do not submit final automatically.",
        ]) + "\n",
        encoding="utf-8",
    )

    state = {
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "protected_best_unchanged": True,
        "protected_best_version": "V6_1_3_gpt55_guarded_endpoint",
        "current_candidate": "V9_4_5_visual_retrieval_fix",
        "experimental_v9_5": {
            "version": VERSION,
            "run_id": RUN_ID,
            "joint": s95.get("joint", 0),
            "result": s95.get("result", 0),
            "tool": s95.get("tool", 0),
            "micro": s95.get("micro", 0),
            "valid": s95.get("valid", 0),
            "retail_trim_events": len(trim_events),
            "not_promoted_reason": "A_medium joint regressed vs V9_4_5 and split valid count drifted to 40.",
        },
        "next_action": "Fix split materialization and process-aware order/retail rerank before B_holdout/final.",
        "final_submitted": False,
    }
    (CODEX / "state" / "v9_candidate_version.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    with (CODEX / "README_STATUS.md").open("a", encoding="utf-8") as f:
        f.write(
            f"\n## V9_5 A_medium Rerank {ts}\n\n"
            f"- Report: `{report}`\n"
            f"- V9_5 A_medium: joint {pct(s95.get('joint',0))}, result {pct(s95.get('result',0))}, tool {pct(s95.get('tool',0))}, micro {pct(s95.get('micro',0))}, valid {s95.get('valid',0)}.\n"
            f"- V9_4_5 A_medium remains stronger on joint: {pct(s945.get('joint',0))}; protected best unchanged.\n"
            f"- Retail trim events: {len(trim_events)}; no completed-run API transport soft failures after no-proxy launcher.\n"
            f"- No validation_B_holdout, no final, no protected best update.\n"
        )

    print(json.dumps({"report": str(report), "rerun": str(rerun), "readiness": str(readiness), "state": str(CODEX / "state" / "v9_candidate_version.json")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
