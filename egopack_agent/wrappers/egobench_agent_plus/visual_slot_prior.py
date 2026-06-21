# -*- coding: utf-8 -*-
"""Selective visual-to-slot prior for V7."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List


CODEX_ROOT = Path(os.environ.get("CODEX_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/codex"))


def _first_json(path: Path) -> Dict[str, Any]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _cache_dirs(scenario: str, task_id: Any) -> List[Path]:
    root = CODEX_ROOT / "visual_cache"
    if not root.exists():
        return []
    return sorted(root.glob(f"{scenario}*_{task_id}"))


def load_visual_slots(scenario: str, state: Dict[str, Any], visual_context: Dict[str, Any] | None = None) -> Dict[str, Any]:
    visual_context = visual_context or {}
    task_id = state.get("task_id")
    cache_dir = None
    cache_id = visual_context.get("cache_id")
    if cache_id:
        c = CODEX_ROOT / "visual_cache" / str(cache_id)
        if c.exists():
            cache_dir = c
    if cache_dir is None:
        dirs = _cache_dirs(scenario, task_id)
        cache_dir = dirs[0] if dirs else None
    data: Dict[str, Any] = {}
    contact_sheet = ""
    if cache_dir:
        data = _first_json(cache_dir / "visual_state.json")
        if (cache_dir / "contact_sheet.jpg").exists():
            contact_sheet = str(cache_dir / "contact_sheet.jpg")
    text = visual_context.get("visual_state_text") or ""
    if not text and cache_dir and (cache_dir / "visual_state.txt").exists():
        try:
            text = (cache_dir / "visual_state.txt").read_text(encoding="utf-8")
        except Exception:
            text = ""
    blob = json.dumps(data, ensure_ascii=False) + "\n" + str(text)
    def pull(pattern: str, limit: int = 5) -> List[str]:
        out = []
        for m in re.findall(pattern, blob, flags=re.I):
            val = re.sub(r"\s+", " ", str(m)).strip(" -:;,.'\"")
            if val and val not in out:
                out.append(val)
            if len(out) >= limit:
                break
        return out
    slots = {
        "restaurant_name": (data.get("restaurant_name") or data.get("restaurant_name_candidates") or [None])[0] if isinstance(data.get("restaurant_name_candidates"), list) else data.get("restaurant_name"),
        "category_candidates": data.get("category_candidates") or pull(r"\b([A-Z][A-Za-z ]{2,30}(?:Pasta|Steaks|Desserts|Drinks|Salads|Meals))\b"),
        "dish_candidates": data.get("dish_candidates") or pull(r"\b([A-Z][A-Za-z' -]{2,40}(?:Chicken|Beef|Pork|Pasta|Rice|Salad|Soup|Steak|Burger|Pizza|Fish|Cutlet))\b"),
        "set_meal_candidates": data.get("set_meal_candidates") or pull(r"\b([A-Z][A-Za-z' -]{2,40}(?:Set Meal|Combo|Platter))\b"),
        "product_candidates": data.get("product_candidates") or [],
        "ingredient_candidates": data.get("ingredient_candidates") or pull(r"\b([a-zA-Z][A-Za-z -]{2,25}(?:flour|pork|chicken|egg|milk|rice|oil|salt|sugar|onion|garlic))\b"),
        "current_recipe_candidates": data.get("current_recipe_candidates") or data.get("recipe_candidates") or [],
        "pointed_entity": data.get("pointed_entity") or "",
        "visible_text": data.get("visible_text") or "",
        "action_sequence": data.get("temporal_events") or data.get("action_sequence") or "",
        "uncertainty": data.get("uncertainty_notes") or "",
        "contact_sheet_path": contact_sheet,
        "source": str(cache_dir) if cache_dir else "",
        "enabled_for_retry": bool(contact_sheet and scenario in {"order", "kitchen"}),
    }
    # Cap lists; visual slots are candidates only.
    for key, val in list(slots.items()):
        if isinstance(val, list):
            slots[key] = [str(x) for x in val if str(x).strip()][:5]
    return slots


def compact_slot_text(slots: Dict[str, Any], scenario: str) -> str:
    if not slots:
        return ""
    keys = ["restaurant_name", "category_candidates", "dish_candidates", "set_meal_candidates", "ingredient_candidates", "current_recipe_candidates", "pointed_entity", "uncertainty"]
    lines = ["Visual-to-slot prior candidates (verify via tools before mutation):"]
    for key in keys:
        val = slots.get(key)
        if val:
            lines.append(f"- {key}: {val}")
    if scenario == "order" and not slots.get("contact_sheet_path"):
        lines.append("- visual_retry: disabled; no contact_sheet for this task")
    return "\n".join(lines)
