# -*- coding: utf-8 -*-
"""Experiment-matrix entry point.

The main autopilot runs the matrix incrementally. This standalone script records
the current matrix state for resume/manual use.
"""

import argparse
import json
import os
from pathlib import Path
import time


CODEX_ROOT = Path(os.environ.get("CODEX_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/codex"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-id", default=f"track2_{time.strftime('%Y%m%d_%H%M%S')}")
    args = parser.parse_args()
    state = CODEX_ROOT / "state" / "best_version.json"
    data = json.load(open(state, encoding="utf-8")) if state.exists() else {}
    print(json.dumps({"run_id": args.run_id, "best": data}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
