#!/usr/bin/env python3
from __future__ import annotations

import glob
import os
import subprocess
from pathlib import Path

CODEX = Path("/home/data-gxu/acm/egolink2026-main/code/track2/codex")
VERSION = "V6_1_gpt55_guarded_endpoint"


def tail(path: Path, n: int = 60) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-n:])
    except Exception as exc:
        return f"[tail_error {type(exc).__name__}: {exc}]"


def main() -> int:
    print("## processes")
    out = subprocess.check_output(["ps", "-eo", "pid=,args="], text=True, errors="replace")
    for line in out.splitlines():
        if "run_gpt55_endpoint_gate" in line or ("track2_multi_agent_plus.py" in line and "gpt-5.5" in line):
            print(line)
    print("\n## latest runs")
    runs = sorted((CODEX / "runs" / VERSION).glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for run in runs[:3]:
        print(run)
    if runs:
        print("\n## logs")
        for path in sorted((runs[0] / "logs").glob("*.log")):
            print(f"\n### {path}")
            print(tail(path, 50))
    print("\n## main log")
    print(tail(CODEX / "logs" / "gpt55_endpoint_gate_latest.log", 80))
    print("\n## reports")
    for path in sorted((CODEX / "reports").glob("02_gpt55_gate_summary_gpt55_endpoint_gate_*.md"), key=lambda p: p.stat().st_mtime, reverse=True)[:3]:
        print(path)
        print(tail(path, 80))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
