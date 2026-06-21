# -*- coding: utf-8 -*-
"""Select a usable Python/conda environment for Track2."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


KEY_MODULES = ["openai", "requests", "pandas", "numpy", "PIL"]
OPTIONAL_MODULES = ["cv2", "torch", "zhipuai"]


def run(cmd, timeout=20):
    try:
        return subprocess.run(
            cmd,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
    except Exception as exc:
        return None


def python_score(py):
    code = (
        "import sys, importlib.util, json; "
        f"mods={KEY_MODULES + OPTIONAL_MODULES!r}; "
        "print(json.dumps({'executable': sys.executable, 'version': sys.version, "
        "'mods': {m: bool(importlib.util.find_spec(m)) for m in mods}}, ensure_ascii=False))"
    )
    proc = run([py, "-c", code])
    if proc is None or proc.returncode != 0:
        return {"python": py, "score": -1, "error": proc.stderr if proc else "failed"}
    try:
        info = json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception as exc:
        return {"python": py, "score": -1, "error": repr(exc), "raw": proc.stdout}
    score = sum(3 for m in KEY_MODULES if info["mods"].get(m)) + sum(1 for m in OPTIONAL_MODULES if info["mods"].get(m))
    return {"python": py, "score": score, **info}


def candidates():
    seen = []
    for py in [os.environ.get("TRACK2_PYTHON"), sys.executable, "python", "python3"]:
        if py and py not in seen:
            seen.append(py)
            yield py
    proc = run(["conda", "env", "list"])
    if proc and proc.returncode == 0:
        for line in proc.stdout.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            prefix = parts[-1]
            if "/" in prefix:
                py = str(Path(prefix) / "bin" / "python")
                if py not in seen:
                    seen.append(py)
                    yield py


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--json-output", required=True)
    args = parser.parse_args()
    results = [python_score(py) for py in candidates()]
    results = sorted(results, key=lambda x: x.get("score", -1), reverse=True)
    best = results[0] if results else {"python": "python", "score": -1}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(f"export TRACK2_PYTHON='{best.get('python', 'python')}'\n")
        f.write(f"export TRACK2_ENV_SCORE='{best.get('score', -1)}'\n")
    with open(args.json_output, "w", encoding="utf-8") as f:
        json.dump({"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "best": best, "candidates": results}, f, ensure_ascii=False, indent=2)
    print(json.dumps({"best": best, "candidates": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
