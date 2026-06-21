#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Refine V7 aggregate intent detection."""

from __future__ import annotations

import shutil
import time
from pathlib import Path


CODEX = Path("/home/data-gxu/acm/egolink2026-main/code/track2/codex")
PATH = CODEX / "wrappers" / "egobench_agent_plus" / "human_process_graph.py"


OLD = '''def _text_needs_aggregate(text: str) -> bool:
    t = str(text or "").lower()
    return bool(re.search(r"\\b(total|tax|payment|payable|price|cost|nutrition|nutritions|calorie|calories|protein|fat|carb|sodium|discount)\\b", t))
'''

NEW = '''def _text_needs_aggregate(text: str) -> bool:
    t = str(text or "").lower()
    if re.search(r"\\b(total|sum|aggregate|overall)\\b", t):
        return True
    if re.search(r"\\b(tax|payment|payable|amount due|checkout|bill)\\b", t):
        return True
    if re.search(r"\\b(total|sum|aggregate|overall)\\s+(nutrition|nutritions|calorie|calories|protein|fat|carb|sodium)", t):
        return True
    return False
'''


def main() -> int:
    text = PATH.read_text(encoding="utf-8")
    if OLD not in text:
        print("pattern_not_found")
        return 1
    dst = CODEX / "backups" / f"v7_aggregate_phrase_{time.strftime('%Y%m%d_%H%M%S')}" / PATH.relative_to(CODEX)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PATH, dst)
    PATH.write_text(text.replace(OLD, NEW, 1), encoding="utf-8")
    print(f"patched {PATH}")
    print(f"backup {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
