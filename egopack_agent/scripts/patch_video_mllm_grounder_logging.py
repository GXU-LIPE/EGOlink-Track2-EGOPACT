#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

TARGET = Path("/home/data-gxu/acm/egolink2026-main/code/track2/codex/wrappers/egobench_agent_plus/video_mllm_grounder.py")


def main() -> None:
    text = TARGET.read_text(encoding="utf-8")
    if "def _log_grounding_event" in text:
        print("already_patched")
        return
    text = text.replace(
        "from typing import Any, Dict, List\n",
        "from typing import Any, Dict, List\nimport time\n",
    )
    insert = r'''

def _log_grounding_event(scenario: str, card: Dict[str, Any]) -> None:
    try:
        version = os.environ.get("TRACK2_RUN_VERSION", "unknown")
        run_id = os.environ.get("TRACK2_RUN_ID", "unknown")
        task_id = os.environ.get("TRACK2_CURRENT_TASK_ID") or card.get("task_id") or "unknown"
        out = CODEX_ROOT / "runs" / str(version) / str(run_id) / "qwen3vl_grounding_hits"
        out.mkdir(parents=True, exist_ok=True)
        event = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "scenario": scenario,
            "task_id": task_id,
            "cache_status": card.get("status") or card.get("path_status") or "unknown",
            "cache_path": card.get("_cache_path", ""),
            "cache_key": card.get("cache_key", ""),
            "teacher": card.get("teacher", ""),
            "top_k_count": len(card.get("top_k_candidates") or []),
            "grounding_failed": card.get("status") == "grounding_failed",
        }
        with (out / f"{task_id}.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception:
        return
'''
    text = text.replace("\ndef _clip(value: Any, limit: int = 420) -> str:\n", insert + "\ndef _clip(value: Any, limit: int = 420) -> str:\n")
    text = text.replace(
        "    if not card:\n        return \"\"\n",
        "    if not card:\n        return \"\"\n    _log_grounding_event(scenario, card)\n",
    )
    text = text.replace(
        "        (\"top_k_candidates\", \"Top-k candidates\"),\n",
        "        (\"top_k_candidates\", \"Top-k candidates\"),\n        (\"cache_key\", \"Cache key\"),\n",
    )
    TARGET.write_text(text, encoding="utf-8")
    print("patched", TARGET)


if __name__ == "__main__":
    main()
