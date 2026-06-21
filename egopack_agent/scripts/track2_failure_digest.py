#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path

CODEX = Path("/home/data-gxu/acm/egolink2026-main/code/track2/codex")
EGO = Path("/home/data-gxu/acm/egolink2026-main/code/track2/EgoBench")
RUN_ID = "gpt55_endpoint_gate_20260617_102324"
VERSION = "V6_1_gpt55_guarded_endpoint"
MODEL = f"gpt-5.5-{VERSION}-{RUN_ID}"


def safe(text: str, limit: int = 12000) -> str:
    text = text.encode("ascii", "replace").decode("ascii")
    return text[:limit]


def load(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def print_log_digest(name: str) -> None:
    path = CODEX / "runs" / VERSION / RUN_ID / "logs" / f"{name}.log"
    print(f"\n## {name} log: {path}")
    text = load(path)
    patterns = [
        r"Final User Response:.*",
        r"Tested Agent:.*",
        r"Guarded Agent:.*",
        r"\s+\[Tool Execution\] Calling:.*",
        r"\s+\[Tool Execution\] Return result:.*",
        r"Completed!.*",
        r"Statistics Summary:.*",
        r"\s+Task 1:.*",
    ]
    lines = []
    for line in text.splitlines():
        if any(re.match(p, line) for p in patterns):
            lines.append(line)
    print(safe("\n".join(lines[-120:])))


def print_eval(name: str) -> None:
    path = EGO / "eval_result" / MODEL / f"{name}_easy_eval.json"
    print(f"\n## {name} eval: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    print(json.dumps(data.get("detailed_results", [])[:2], ensure_ascii=True, indent=2))
    print(json.dumps(data.get("micro_tool_stats", {}), ensure_ascii=True, indent=2))


def main() -> None:
    for name in ["order1", "kitchen2"]:
        print_log_digest(name)
        print_eval(name)


if __name__ == "__main__":
    main()
