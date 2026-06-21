#!/usr/bin/env python3
from __future__ import annotations

import time
from pathlib import Path

CODEX = Path("/home/data-gxu/acm/egolink2026-main/code/track2/codex")
readme = CODEX / "README_STATUS.md"
stamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
block = f"""

## GPT-5.5 bridge status ({stamp})

- Current status: blocked by OpenAI authentication, not by remote network.
- Remote health check to local bridge: OK at `http://127.0.0.1:17901/health`.
- API call result: OpenAI returned `401 invalid_api_key`; key value was not logged.
- Valid GPT-5.5 Track2 gate metrics: not available yet.
- Cleanup: duplicate bridge gate runs were stopped.
- Latest blocker report: `reports/gpt55_bridge_blocker_20260617_004602.md`.
- Rerun after replacing key: `bash scripts/run_gpt55_bridge_gate.sh`.
- Final submission: not submitted.
"""
old = readme.read_text(encoding="utf-8") if readme.exists() else "# Track2 Codex Status\n"
if "## GPT-5.5 bridge status" in old:
    old = old.split("## GPT-5.5 bridge status", 1)[0].rstrip() + "\n"
readme.write_text(old.rstrip() + block + "\n", encoding="utf-8")
print(readme)
