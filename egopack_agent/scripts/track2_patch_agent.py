# -*- coding: utf-8 -*-
"""Reserved patch helper for Track2 official-code edits.

Current phase avoids official-code edits. This script records that policy and
can later be extended to backup files and write unified diffs before patching.
"""

import argparse
import os
from pathlib import Path
import time


CODEX_ROOT = Path(os.environ.get("CODEX_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/codex"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = CODEX_ROOT / "reports" / f"patch_policy_{time.strftime('%Y%m%d_%H%M%S')}.md"
    text = "# Patch Policy\n\nNo official EgoBench files patched in current phase. Copied runners and PYTHONPATH wrappers are used.\n"
    if not args.dry_run:
        report.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
