# -*- coding: utf-8 -*-
"""Write a compact inventory JSON for Track2 automation."""

import argparse
import json
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path


def run(cmd, cwd=None, timeout=30):
    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            shell=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        return {
            "cmd": cmd,
            "cwd": cwd,
            "exit_code": proc.returncode,
            "stdout": proc.stdout[-8000:],
            "stderr": proc.stderr[-8000:],
            "elapsed": round(time.time() - start, 3),
        }
    except Exception as exc:
        return {"cmd": cmd, "cwd": cwd, "exit_code": -1, "error": repr(exc), "elapsed": round(time.time() - start, 3)}


def count_files(path, pattern):
    root = Path(path)
    if not root.exists():
        return 0
    return sum(1 for _ in root.rglob(pattern))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ego-root", required=True)
    parser.add_argument("--codex-root", required=True)
    args = parser.parse_args()

    ego = Path(args.ego_root)
    codex = Path(args.codex_root)
    data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "user": os.environ.get("USER", ""),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "ego_root": str(ego),
        "codex_root": str(codex),
        "report": args.report,
        "commands": {},
        "counts": {
            "scenario_files": count_files(ego / "scenarios", "*.json"),
            "tool_json_files": count_files(ego / "tools", "*_tools.json"),
            "video_files": count_files(ego / "videos", "*"),
            "result_files": count_files(ego / "results", "*.json"),
            "eval_files": count_files(ego / "eval_result", "*.json"),
        },
        "final_files": {},
    }
    for rel in [
        "scenarios/final/retail6.json",
        "scenarios/final/retail10.json",
        "scenarios/final/kitchen4.json",
        "scenarios/final/restaurant5.json",
        "scenarios/final/order2.json",
    ]:
        data["final_files"][rel] = (ego / rel).exists()

    data["commands"]["python"] = run("python -c \"import sys; print(sys.executable); print(sys.version)\"")
    data["commands"]["conda_env_list"] = run("conda env list 2>/dev/null || true")
    data["commands"]["nvidia_smi"] = run("nvidia-smi", timeout=20)
    data["commands"]["disk"] = run(f"df -h {codex} {ego}", timeout=20)
    data["commands"]["git"] = run("git rev-parse HEAD; git status --short", cwd=str(ego), timeout=20)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
