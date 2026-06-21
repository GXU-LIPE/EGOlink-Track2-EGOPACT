# -*- coding: utf-8 -*-
"""V16 slot resolver utilities used by prompt/policy code."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List


CODEX_ROOT = Path(os.environ.get("CODEX_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/codex"))


def load_v16_lexicon() -> Dict[str, Any]:
    path = Path(os.environ.get("TRACK2_V16_DISTILL_DIR") or (CODEX_ROOT / "gt_distill_v16")) / "scenario_entity_lexicon.json"
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return {}


def top_slot_candidates(scenario: str, slot: str, limit: int = 12) -> List[str]:
    lex = load_v16_lexicon()
    values = lex.get(scenario, {}).get(slot, [])
    return [str(x) for x in values[:limit]]
