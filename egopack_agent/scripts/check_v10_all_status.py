#!/usr/bin/env python3
import json
import pathlib
import re
import subprocess

CODEX = pathlib.Path("/home/data-gxu/acm/egolink2026-main/code/track2/codex")
EGO = pathlib.Path("/home/data-gxu/acm/egolink2026-main/code/track2/EgoBench")
SCENARIOS = ["retail6", "retail10", "kitchen4", "restaurant5", "order2"]


def run(cmd):
    return subprocess.run(cmd, shell=True, text=True, capture_output=True)


print("--- active ---")
ps = run("ps -o pid,ppid,stat,etime,cmd -C python3 -C bash")
for line in ps.stdout.splitlines():
    if (
        "run_v10_final_full.sh" in line
        or "run_v10_final_remaining_parallel.sh" in line
        or "track2_multi_agent_plus.py" in line
        or line.strip().startswith("PID")
    ):
        print(line)

print("--- scenario progress ---")
for tag in SCENARIOS:
    log = CODEX / "logs" / f"v10_final_full_{tag}_20260618_1940.log"
    if not log.exists():
        log = CODEX / "logs" / f"v10_final_parallel_{tag}_20260618_2010.log"
    if not log.exists():
        log = CODEX / "logs" / f"v10_final_parallel_{tag}_20260618_2032.log"
    if not log.exists():
        print(tag, "no_log")
        continue
    text = log.read_text(encoding="utf-8", errors="ignore")
    matches = re.findall(r"=+ Scenario [^:]+: ([0-9]+) =+", text)
    err = re.findall(r"(?i)(traceback|exception|429|rate limit|readtimeout|connectionerror|api_error)", text)
    print(tag, "last_task", matches[-1] if matches else "-", "tasks_seen", len(matches), "lines", text.count("\n"), "hard_errors", len(err))

print("--- result counts ---")
root = EGO / "results" / "V10_full_memory_final_candidate_draft"
print("root_exists", root.exists())
for p in sorted(root.glob("*_easy.json")):
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        print(p.name, len(data), p.stat().st_size)
    except Exception as exc:
        print(p.name, type(exc).__name__, p.stat().st_size)
