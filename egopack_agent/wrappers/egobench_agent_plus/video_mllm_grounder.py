# -*- coding: utf-8 -*-
"""Qwen3-VL video grounding teacher support for Track2 V12.

The service agent never calls Qwen3-VL directly. This module only reads cached
JSON grounding cards produced by scripts/build_v12_qwen3vl_grounding.py and
formats them as compact prompt evidence.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List
import time

CODEX_ROOT = Path(os.environ.get("CODEX_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/codex"))
DEFAULT_CACHE_DIR = CODEX_ROOT / "visual_cache_v12" / "qwen3vl_grounding"


def is_enabled() -> bool:
    return os.environ.get("TRACK2_ENABLE_QWEN3VL_GROUNDING", "0") == "1"


def _cache_dir() -> Path:
    raw = os.environ.get("TRACK2_QWEN3VL_GROUNDING_DIR", "")
    if not raw:
        return DEFAULT_CACHE_DIR
    path = Path(raw)
    return path if path.is_absolute() else CODEX_ROOT / path


def _video_cache_dir() -> Path:
    raw = os.environ.get("TRACK2_QWEN3VL_VIDEO_GROUNDING_DIR", "")
    if not raw:
        return CODEX_ROOT / "visual_cache_v12" / "qwen3vl_grounding_by_video"
    path = Path(raw)
    return path if path.is_absolute() else CODEX_ROOT / path


def _candidate_ids(scenario: str) -> List[str]:
    task_id = os.environ.get("TRACK2_CURRENT_TASK_ID") or "1"
    spec = os.environ.get("TRACK2_CURRENT_SCENARIO_SPEC") or scenario
    return [
        f"{spec}_{task_id}",
        f"{scenario}_{task_id}",
        f"{spec}_task{task_id}",
    ]


def _has_candidates(card: Dict[str, Any]) -> bool:
    return bool(card.get("top_k_candidates"))


def _load_json(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    data["_cache_path"] = str(path)
    return data


def _load_video_fallback(primary: Dict[str, Any], scenario: str) -> Dict[str, Any]:
    video_path = str(primary.get("video_path") or "")
    if not video_path:
        return {}
    video_dir = _video_cache_dir()
    if not video_dir.exists():
        return {}
    best: Dict[str, Any] = {}
    for path in sorted(video_dir.glob("*.json")):
        try:
            data = _load_json(path)
        except Exception:
            continue
        if str(data.get("video_path") or "") != video_path:
            continue
        if scenario and data.get("scenario") not in {scenario, None, ""}:
            # Same video can be shared, but avoid cross-scenario fallback
            # unless no better card exists.
            best = best or data
            continue
        best = data
        if _has_candidates(data):
            break
    if not best:
        return {}
    merged = dict(primary)
    for key in [
        "scene_summary",
        "visible_text",
        "visible_products",
        "visible_dishes",
        "visible_ingredients",
        "pointed_or_held_objects",
        "relative_location_objects",
        "category_country_brand_taste_clues",
        "restaurant_menu_order_clues",
        "top_k_candidates",
        "uncertainty_notes",
    ]:
        if key == "uncertainty_notes":
            notes = []
            if isinstance(primary.get(key), list):
                notes.extend(primary.get(key) or [])
            elif primary.get(key):
                notes.append(str(primary.get(key)))
            notes.append("task-specific grounding card lacked top_k_candidates; using same-video Qwen3-VL fallback card as visual prior.")
            if isinstance(best.get(key), list):
                notes.extend(best.get(key) or [])
            elif best.get(key):
                notes.append(str(best.get(key)))
            merged[key] = notes[:12]
        elif not primary.get(key) or (key == "top_k_candidates" and not _has_candidates(primary)):
            merged[key] = best.get(key)
    merged["_cache_path"] = primary.get("_cache_path", "")
    merged["_video_fallback_cache_path"] = best.get("_cache_path", "")
    merged["_video_fallback_used"] = True
    merged["_task_card_status"] = primary.get("status", "")
    merged["status"] = "qwen3vl_success" if _has_candidates(merged) else primary.get("status", "grounding_failed")
    merged["teacher"] = "qwen3vl"
    return merged


def load_grounding_card(scenario: str) -> Dict[str, Any]:
    if not is_enabled():
        return {}
    for cid in _candidate_ids(scenario):
        path = _cache_dir() / f"{cid}.json"
        if path.exists():
            try:
                data = _load_json(path)
                if not _has_candidates(data):
                    fallback = _load_video_fallback(data, scenario)
                    if fallback:
                        return fallback
                return data
            except Exception as exc:
                return {"status": "cache_parse_error", "cache_id": cid, "error": type(exc).__name__}
    return {"status": "missing", "cache_ids_checked": _candidate_ids(scenario)}



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
            "video_fallback_used": bool(card.get("_video_fallback_used")),
            "video_fallback_cache_path": card.get("_video_fallback_cache_path", ""),
            "task_card_status": card.get("_task_card_status", ""),
        }
        with (out / f"{task_id}.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception:
        return

def _clip(value: Any, limit: int = 420) -> str:
    if isinstance(value, (list, tuple)):
        text = "; ".join(str(x) for x in value if str(x).strip())
    elif isinstance(value, dict):
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    else:
        text = str(value or "")
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def build_qwen3vl_grounding_prompt(scenario: str) -> str:
    if not is_enabled():
        return ""
    card = load_grounding_card(scenario)
    if not card:
        return ""
    _log_grounding_event(scenario, card)
    status = card.get("status") or card.get("path_status") or "unknown"
    lines = ["[V12 Qwen3-VL Video Grounding Teacher]"]
    lines.append("- Role: visual grounding evidence only. It must not directly choose tool actions or override tool observations.")
    lines.append(f"- Cache status: {status}; task_id: {card.get('task_id','')}; scenario: {card.get('scenario', scenario)}")
    if card.get("_cache_path"):
        lines.append(f"- Cache path: {card['_cache_path']}")
    if card.get("_video_fallback_used"):
        lines.append(f"- Same-video fallback used: yes; fallback path: {card.get('_video_fallback_cache_path','')}")
        lines.append(f"- Original task-card status: {card.get('_task_card_status','')}")
    for key, label in [
        ("scene_summary", "Scene"),
        ("visible_text", "Readable text"),
        ("visible_products", "Visible products"),
        ("visible_dishes", "Visible dishes"),
        ("visible_ingredients", "Visible ingredients"),
        ("pointed_or_held_objects", "Pointed/held objects"),
        ("relative_location_objects", "Relative-location objects"),
        ("category_country_brand_taste_clues", "Category/country/brand/taste clues"),
        ("restaurant_menu_order_clues", "Restaurant/menu/order clues"),
        ("top_k_candidates", "Top-k candidates"),
        ("cache_key", "Cache key"),
        ("uncertainty_notes", "Uncertainty"),
    ]:
        value = card.get(key)
        if value:
            lines.append(f"- {label}: {_clip(value)}")
    if status == "missing":
        lines.append("- No cached Qwen3-VL card was available. Use V10 memory and retrieval tools; do not ask the user for visual labels.")
    elif not card.get("top_k_candidates"):
        lines.append("- This Qwen3-VL card has no top-k candidates. Treat it as uncertainty evidence only, then use memory/retrieval tools to ground entities.")
    else:
        lines.append("- Use this card to narrow candidates, then verify with official tools before mutation or aggregate computation.")
    return "\n".join(lines)
