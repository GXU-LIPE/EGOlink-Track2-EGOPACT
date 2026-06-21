# -*- coding: utf-8 -*-
"""Load optional visual state cache."""

import json
import os
from pathlib import Path
from typing import Dict


CODEX_ROOT = Path(os.environ.get("CODEX_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/codex"))


def load_visual_state(video_id: str) -> Dict:
    path = CODEX_ROOT / "visual_cache" / video_id / "visual_state.json"
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}
