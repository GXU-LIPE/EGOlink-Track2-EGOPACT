#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run V7 gate runner unbuffered so timeout diagnostics keep logs."""

from __future__ import annotations

import shutil
import time
from pathlib import Path


CODEX = Path("/home/data-gxu/acm/egolink2026-main/code/track2/codex")
PATH = CODEX / "scripts" / "run_human_prior_gate.sh"


OLD = '''  timeout "$SCENARIO_TIMEOUT" "$PYTHON_BIN" "$CODEX/runners/track2_multi_agent_plus.py" \\
'''

NEW = '''  timeout "$SCENARIO_TIMEOUT" "$PYTHON_BIN" -u "$CODEX/runners/track2_multi_agent_plus.py" \\
'''


def main() -> int:
    text = PATH.read_text(encoding="utf-8")
    if NEW in text:
        print("already_patched")
        return 0
    if OLD not in text:
        print("pattern_not_found")
        return 1
    dst = CODEX / "backups" / f"v7_unbuffered_{time.strftime('%Y%m%d_%H%M%S')}" / PATH.relative_to(CODEX)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PATH, dst)
    PATH.write_text(text.replace(OLD, NEW, 1), encoding="utf-8")
    print(f"patched {PATH}")
    print(f"backup {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
