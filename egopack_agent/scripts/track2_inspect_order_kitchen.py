#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

CODEX = Path("/home/data-gxu/acm/egolink2026-main/code/track2/codex")
EGO = Path("/home/data-gxu/acm/egolink2026-main/code/track2/EgoBench")
RUN_ID = "gpt55_endpoint_gate_20260617_102324"
VERSION = "V6_1_gpt55_guarded_endpoint"


def safe(text: str, limit: int = 20000) -> str:
    return text.encode("ascii", "replace").decode("ascii")[:limit]


def show(path: Path, limit: int = 12000) -> None:
    print(f"\n## {path}")
    if not path.exists():
        print("MISSING")
        return
    text = path.read_text(encoding="utf-8", errors="replace")
    print(safe(text, limit))


def main() -> None:
    for path in [
        EGO / "scenarios" / "final" / "order1.json",
        EGO / "tools" / "order" / "order_tools.json",
        EGO / "tools" / "kitchen" / "kitchen_tools.json",
        CODEX / "visual_cache" / "order1_1" / "visual_state.txt",
        CODEX / "visual_cache" / "order1_1" / "visual_state.json",
        CODEX / "visual_cache" / "order1_1" / "frames_manifest.json",
        CODEX / "visual_cache" / "order1_1" / "contact_sheet.jpg",
        CODEX / "runs" / VERSION / RUN_ID / "logs" / "order1.log",
        CODEX / "runs" / VERSION / RUN_ID / "logs" / "kitchen2.log",
    ]:
        if path.suffix.lower() in {".jpg", ".png"}:
            print(f"\n## {path}")
            print("exists", path.exists(), "size", path.stat().st_size if path.exists() else 0)
        else:
            show(path)


if __name__ == "__main__":
    main()
