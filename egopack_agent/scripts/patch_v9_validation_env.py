#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
import time


CODEX = Path("/home/data-gxu/acm/egolink2026-main/code/track2/codex")
SCRIPT = CODEX / "scripts" / "run_v8_validation.py"
BACKUP = CODEX / "backups" / f"v9_validation_env_{time.strftime('%Y%m%d_%H%M%S')}" / "scripts" / "run_v8_validation.py"


def main():
    BACKUP.parent.mkdir(parents=True, exist_ok=True)
    BACKUP.write_bytes(SCRIPT.read_bytes())
    text = SCRIPT.read_text(encoding="utf-8-sig")
    needle = "        'TRACK2_ENABLE_MULTICANDIDATE': '1' if 'multicandidate' in version or 'top1' in version else env.get('TRACK2_ENABLE_MULTICANDIDATE','0'),\n"
    insert = needle
    insert += "        'TRACK2_ENABLE_EVALUATOR_AWARENESS': '1' if version.startswith('V9_') else env.get('TRACK2_ENABLE_EVALUATOR_AWARENESS','0'),\n"
    insert += "        'TRACK2_ENABLE_V9_SOFT_GUARD': '1' if version.startswith('V9_2') or version.startswith('V9_3') or version.startswith('V9_4') or version.startswith('V9_5') else env.get('TRACK2_ENABLE_V9_SOFT_GUARD','0'),\n"
    insert += "        'TRACK2_ENABLE_DEEPSEEK_CROSSCHECK': '0' if version.startswith('V9_1') or version.startswith('V9_2') else ('1' if 'deepseek' in version else env.get('TRACK2_ENABLE_DEEPSEEK_CROSSCHECK','0')),\n"
    if "TRACK2_ENABLE_EVALUATOR_AWARENESS" not in text:
        text = text.replace(needle, insert)
    SCRIPT.write_text(text, encoding="utf-8")
    print({"patched": True, "backup": str(BACKUP)})


if __name__ == "__main__":
    main()
