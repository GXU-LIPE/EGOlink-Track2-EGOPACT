#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import pathlib
import subprocess

CODEX = pathlib.Path("/home/data-gxu/acm/egolink2026-main/code/track2/codex")
RUN_ID = "V12_qwen3vl_prior_all_modules_val41_parallel_20260619_170302"
VERSION = "V12_official_style_qwen3vl_memory"
run_dir = CODEX / "runs" / VERSION / RUN_ID
partial = run_dir / "run_items.partial.json"
final_items = run_dir / "run_items.json"
eval_summary = run_dir / "eval_summary.json"
qwen_audit = run_dir / "qwen3vl_prompt_hit_audit.json"
latest = CODEX / "state" / "latest_v12_val41_parallel.json"
report = CODEX / "reports" / f"V12_VAL41_PARALLEL_QWEN3VL_MEMORY_{RUN_ID}.md"

out = {
    "run_id": RUN_ID,
    "partial_exists": partial.exists(),
    "final_items_exists": final_items.exists(),
    "eval_summary_exists": eval_summary.exists(),
    "qwen_audit_exists": qwen_audit.exists(),
    "latest_exists": latest.exists(),
    "report_exists": report.exists(),
}
for name, path in [("partial", partial), ("final_items", final_items)]:
    if path.exists():
        try:
            items = json.loads(path.read_text(encoding="utf-8"))
            out[f"{name}_count"] = len(items)
            out[f"{name}_items"] = [
                {
                    "spec": x.get("spec"),
                    "rc": x.get("returncode"),
                    "seconds": x.get("seconds"),
                    "error": x.get("error", ""),
                }
                for x in sorted(items, key=lambda y: y.get("spec", ""))
            ]
        except Exception as exc:
            out[f"{name}_error"] = f"{type(exc).__name__}: {exc}"
if eval_summary.exists():
    try:
        out["summary"] = json.loads(eval_summary.read_text(encoding="utf-8")).get("summary")
    except Exception as exc:
        out["summary_error"] = f"{type(exc).__name__}: {exc}"
if qwen_audit.exists():
    try:
        audit = json.loads(qwen_audit.read_text(encoding="utf-8"))
        out["qwen_audit"] = {
            "event_count": audit.get("event_count"),
            "task_count": audit.get("task_count"),
            "events_with_top_k": audit.get("events_with_top_k"),
            "video_fallback_events": audit.get("video_fallback_events"),
            "missing_or_failed_events": audit.get("missing_or_failed_events"),
            "status_counts": audit.get("status_counts"),
        }
    except Exception as exc:
        out["qwen_audit_error"] = f"{type(exc).__name__}: {exc}"
try:
    ps = subprocess.run(
        ["pgrep", "-af", "run_v12_val41_parallel.py|track2_multi_agent_plus.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=5,
    )
    out["processes"] = ps.stdout.strip().splitlines()
except Exception as exc:
    out["process_error"] = f"{type(exc).__name__}: {exc}"

print(json.dumps(out, ensure_ascii=False, indent=2))
