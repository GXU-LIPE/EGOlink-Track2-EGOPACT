#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path


CODEX = Path("/home/data-gxu/acm/egolink2026-main/code/track2/codex")


def matching_processes():
    out = subprocess.check_output(["ps", "-eo", "pid=,args="], text=True, errors="replace")
    rows = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        pid_s, _, args = line.partition(" ")
        try:
            pid = int(pid_s)
        except ValueError:
            continue
        if pid == os.getpid():
            continue
        if "run_gpt55_bridge_gate.sh" in args or ("track2_multi_agent_plus.py" in args and "gpt-5.5" in args):
            rows.append((pid, args))
    return rows


def stop_processes(rows):
    stopped = []
    for pid, args in rows:
        try:
            os.kill(pid, signal.SIGTERM)
            stopped.append({"pid": pid, "signal": "TERM", "args": args[:240]})
        except ProcessLookupError:
            pass
        except Exception as exc:
            stopped.append({"pid": pid, "error": type(exc).__name__, "args": args[:240]})
    time.sleep(2)
    alive = {pid for pid, _ in matching_processes()}
    for pid, args in rows:
        if pid in alive:
            try:
                os.kill(pid, signal.SIGKILL)
                stopped.append({"pid": pid, "signal": "KILL", "args": args[:240]})
            except Exception as exc:
                stopped.append({"pid": pid, "kill_error": type(exc).__name__, "args": args[:240]})
    return stopped


def main():
    CODEX.joinpath("reports").mkdir(parents=True, exist_ok=True)
    CODEX.joinpath("state").mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    rows = matching_processes()
    stopped = stop_processes(rows)
    report = CODEX / "reports" / f"gpt55_bridge_blocker_{ts}.md"
    report.write_text(
        "\n".join(
            [
                "# GPT-5.5 Bridge Blocker",
                "",
                f"- generated_at: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}",
                "- status: blocked_invalid_openai_key",
                "- openai_key_present: yes",
                "- api_error: OpenAI returned 401 invalid_api_key",
                "- key_value_logged: no",
                "- local_bridge: reachable from remote at http://127.0.0.1:17901/health",
                "- reverse_tunnel: reachable",
                "- gate_status: stopped; no valid GPT-5.5 validation metrics generated",
                "- success_rate: unavailable because the API rejected authentication before model inference",
                "- final_submission: not submitted",
                "",
                "## Cleanup",
                "",
                f"- matched_processes_before_cleanup: {len(rows)}",
                f"- stop_records: {len(stopped)}",
                "",
                "## Next",
                "",
                "Provide a valid OPENAI_API_KEY in codex/state/.openai_env or local bridge env, then rerun scripts/run_gpt55_bridge_gate.sh through the bridge.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    state = {
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "blocked_invalid_openai_key",
        "openai_key_present": True,
        "key_value_logged": False,
        "local_bridge_reachable": True,
        "valid_gate_metrics": False,
        "report": str(report.relative_to(CODEX)),
    }
    (CODEX / "state" / "gpt55_bridge_blocker.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(report)
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
