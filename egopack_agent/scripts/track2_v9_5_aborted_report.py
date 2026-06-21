#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import time
from pathlib import Path

CODEX = Path("/home/data-gxu/acm/egolink2026-main/code/track2/codex")
RUN_ID = "V9_5_memory_deepseek_rerank_validation_A_medium_20260618_1202"
VERSION = "V9_5_memory_deepseek_rerank"
RUN_DIR = CODEX / "runs" / VERSION / RUN_ID

ts = time.strftime("%Y%m%d_%H%M%S")
log = RUN_DIR / "logs" / "kitchen1.log"
timeout_count = 0
soft_failure_count = 0
snippet = ""
if log.exists():
    text = log.read_text(encoding="utf-8", errors="replace")
    timeout_count = text.count("ReadTimeout")
    soft_failure_count = text.count("Direct API Soft Failure")
    snippet = "\n".join(text.splitlines()[-40:])

report = CODEX / "reports" / f"V9_5_ABORTED_TRANSPORT_{ts}.md"
report.write_text(
    "\n".join([
        f"# V9_5 Aborted Transport {ts}",
        "",
        f"- run_id: `{RUN_ID}`",
        f"- version: `{VERSION}`",
        "- final_submission: not submitted",
        "- protected_best_updated: no",
        "- status: aborted manually because repeated endpoint read timeouts polluted the trajectory before evaluation",
        f"- ReadTimeout count in kitchen1.log: {timeout_count}",
        f"- Direct API Soft Failure count in kitchen1.log: {soft_failure_count}",
        "",
        "## Action",
        "",
        "- Stopped the run before full A_medium scoring.",
        "- Next launch should raise `TRACK2_READ_TIMEOUT` and `TRACK2_API_MAX_RETRIES` and healthcheck the endpoint first.",
        "",
        "## Tail Snippet",
        "",
        "```text",
        snippet[-3000:],
        "```",
    ]) + "\n",
    encoding="utf-8",
)
print(json.dumps({"report": str(report), "timeout_count": timeout_count, "soft_failure_count": soft_failure_count}, ensure_ascii=False, indent=2))
