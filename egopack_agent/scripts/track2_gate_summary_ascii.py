#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

CODEX = Path("/home/data-gxu/acm/egolink2026-main/code/track2/codex")
EGO = Path("/home/data-gxu/acm/egolink2026-main/code/track2/EgoBench")
VERSION = os.environ.get("TRACK2_SUMMARY_VERSION", "V6_1_gpt55_guarded_endpoint")


def safe(s: str) -> str:
    return s.encode("ascii", "replace").decode("ascii")


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception as exc:
        return {"_error": type(exc).__name__, "_path": str(path)}


def main():
    out = subprocess.check_output(["ps", "-eo", "pid=,args="], text=True, errors="replace")
    print("processes")
    for line in out.splitlines():
        if "run_gpt55_endpoint_gate" in line or ("track2_multi_agent_plus.py" in line and "gpt-5.5" in line):
            print(safe(line[:240]))

    runs = sorted((CODEX / "runs" / VERSION).glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
    run = runs[0] if runs else None
    print("latest_run", safe(str(run) if run else "none"))
    if not run:
        return
    run_id = run.name
    model_name = f"gpt-5.5-{VERSION}-{run_id}"
    print("model_name", model_name)
    for scenario in ["retail9", "kitchen2", "restaurant4", "order1"]:
        log = run / "logs" / f"{scenario}.log"
        exists = log.exists()
        completed = False
        tool_calls = None
        rounds = None
        if exists:
            text = log.read_text(encoding="utf-8", errors="replace")
            completed = "Completed! Results saved" in text
            import re

            m = re.findall(r"Task 1: (\\d+) dialogue rounds.*?(\\d+) tool calls", text)
            if m:
                rounds, tool_calls = m[-1]
        print(f"log {scenario} exists={exists} completed={completed} rounds={rounds} tool_calls={tool_calls}")
    eval_dir = EGO / "eval_result" / model_name
    print("eval_dir_exists", eval_dir.exists(), safe(str(eval_dir)))
    for path in sorted(eval_dir.glob("*_eval.json")):
        data = load_json(path)
        keys = {}
        if isinstance(data, dict):
            for k in [
                "tool_based_success_rate",
                "result_based_success_rate",
                "joint_success_rate",
                "micro_tool_call_accuracy",
                "avg_tool_calls",
            ]:
                if k in data:
                    keys[k] = data[k]
            # fallback: print top scalar fields
            if not keys:
                for k, v in data.items():
                    if isinstance(v, (int, float, str, bool)) and len(keys) < 8:
                        keys[k] = v
        print("eval", path.name, json.dumps(keys, ensure_ascii=True))
    reports = sorted(CODEX.glob(f"reports/02_gpt55_gate_summary_{run_id}.md"), key=lambda p: p.stat().st_mtime, reverse=True)
    for report in reports[:1]:
        print("report", safe(str(report)))
        print(safe(report.read_text(encoding="utf-8", errors="replace")[:3000]))


if __name__ == "__main__":
    main()
