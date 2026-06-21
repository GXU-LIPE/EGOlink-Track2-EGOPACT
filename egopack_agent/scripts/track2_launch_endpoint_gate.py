#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

CODEX = Path("/home/data-gxu/acm/egolink2026-main/code/track2/codex")
CODEX.joinpath("logs").mkdir(parents=True, exist_ok=True)
CODEX.joinpath("state").mkdir(parents=True, exist_ok=True)

run_id = os.environ.get("TRACK2_RUN_ID") or "gpt55_endpoint_gate_" + time.strftime("%Y%m%d_%H%M%S")
log_path = CODEX / "logs" / "gpt55_endpoint_gate_latest.log"
pid_path = CODEX / "state" / "gpt55_endpoint_gate.pid"

env = os.environ.copy()
env.setdefault("TRACK2_MAX_TURNS", "6")
env["TRACK2_RUN_ID"] = run_id

with log_path.open("wb") as log:
    proc = subprocess.Popen(
        ["bash", str(CODEX / "scripts" / "run_gpt55_endpoint_gate.sh")],
        cwd=str(CODEX),
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

pid_path.write_text(str(proc.pid) + "\n", encoding="utf-8")
(CODEX / "state" / "latest_gpt55_endpoint_gate_launch.json").write_text(
    "{\n"
    f'  "run_id": "{run_id}",\n'
    f'  "pid": {proc.pid},\n'
    f'  "log": "logs/gpt55_endpoint_gate_latest.log",\n'
    f'  "started_at": "{time.strftime("%Y-%m-%dT%H:%M:%S%z")}"\n'
    "}\n",
    encoding="utf-8",
)
print(f"started run_id={run_id} pid={proc.pid} log={log_path}")
