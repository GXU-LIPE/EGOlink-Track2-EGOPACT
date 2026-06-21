#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build cached GPT-5.5 visual_state from Track2 contact sheets."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict


CODEX_ROOT = Path(os.environ.get("CODEX_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/codex"))
EGO_ROOT = Path(os.environ.get("EGO_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/EgoBench"))
sys.path.insert(0, str(CODEX_ROOT / "wrappers"))
sys.path.insert(0, str(CODEX_ROOT))

from egobench_agent_plus.openai_gpt55_adapter import call_openai_gpt55  # noqa: E402
from track2_extract_video_frames import process_task  # noqa: E402


FIELDS = [
    "video_path",
    "path_status",
    "scenario",
    "scene_summary",
    "visible_text",
    "visible_objects",
    "pointed_or_held_objects",
    "temporal_events",
    "restaurant_name_candidates",
    "category_candidates",
    "dish_candidates",
    "set_meal_candidates",
    "product_candidates",
    "ingredient_candidates",
    "spatial_relations",
    "uncertainty_notes",
    "evidence_frames",
]


def _empty_state(manifest: Dict[str, Any], reason: str) -> Dict[str, Any]:
    return {
        "video_path": manifest.get("video_path", ""),
        "path_status": manifest.get("path_status", reason),
        "scenario": manifest.get("scenario", ""),
        "scene_summary": "",
        "visible_text": [],
        "visible_objects": [],
        "pointed_or_held_objects": [],
        "temporal_events": [],
        "restaurant_name_candidates": [],
        "category_candidates": [],
        "dish_candidates": [],
        "set_meal_candidates": [],
        "product_candidates": [],
        "ingredient_candidates": [],
        "spatial_relations": [],
        "uncertainty_notes": [reason],
        "evidence_frames": manifest.get("frames", []),
    }


def _task_context(scenario: str, number: int, task_index: int) -> Dict[str, Any]:
    if os.environ.get("TRACK2_FINAL_EVAL", "0") == "1":
        return {
            "scenario": f"{scenario}{number}",
            "task_index": task_index,
            "final_compliant": True,
            "note": "Final mode: scenarios/final JSON metadata is not read for service-agent visual_state.",
        }
    path = EGO_ROOT / "scenarios" / "final" / f"{scenario}{number}.json"
    tasks = json.loads(path.read_text(encoding="utf-8"))
    task = tasks[max(0, task_index - 1)]
    return {
        "Instruction": task.get("Instruction", ""),
        "image_description": task.get("image_description", ""),
        "analysis": task.get("analysis", task.get("Analysis", task.get("task_analysis", ""))),
        "image_path": task.get("image_path", ""),
    }


def build_visual_state(scenario: str, number: int, task_index: int, force: bool = False) -> Dict[str, Any]:
    if os.environ.get("TRACK2_FINAL_EVAL", "0") == "1":
        state = _empty_state({"scenario": scenario, "frames": []}, "final_mode_no_direct_scenario_json_access")
        state["final_compliant_no_direct_final_json"] = True
        return state
    manifest = process_task(scenario, number, task_index, force=False)
    cache_dir = CODEX_ROOT / "visual_cache" / manifest["cache_id"]
    out_json = cache_dir / "visual_state.json"
    out_txt = cache_dir / "visual_state.txt"
    if out_json.exists() and not force:
        return json.loads(out_json.read_text(encoding="utf-8"))
    if not os.environ.get("OPENAI_API_KEY"):
        state = _empty_state(manifest, "OPENAI_API_KEY_missing_visual_state_not_generated")
    elif not manifest.get("contact_sheet_created"):
        state = _empty_state(manifest, "contact_sheet_missing")
    else:
        context = _task_context(scenario, number, task_index)
        prompt = f"""You are extracting visual evidence for EgoBench Track2.
Return only JSON with these fields: {FIELDS}.
Scenario: {scenario}
Task context: {json.dumps(context, ensure_ascii=False)}

Focus:
- order: restaurant name, menu categories, dish/set meal names, selected/replaced/removed/current order items.
- kitchen: cooking step, ingredients, tools/containers/fridge/stove/tray, action sequence.
- retail: product text/brand, shelf position, color/shape, pointed item.
- restaurant: dish/set meal/menu text, table item, selected item.
Use uncertainty_notes for anything unclear. Do not invent text that is not visible."""
        text, _, _ = call_openai_gpt55(
            [{"role": "user", "content": prompt}],
            agent_type="visual_state",
            service_model_name=os.environ.get("TRACK2_OPENAI_MODEL", "gpt-5.5"),
            contact_sheet_path=manifest.get("contact_sheet"),
            force_high_effort=True,
        )
        try:
            state = json.loads(text)
            if isinstance(state, list):
                state = {"scene_summary": str(state)}
            if not isinstance(state, dict):
                raise ValueError("not_dict")
        except Exception:
            state = _empty_state(manifest, "visual_state_parse_failed")
            state["raw_model_output"] = text[:2000]
        for field in FIELDS:
            state.setdefault(field, [] if field.endswith("candidates") or field in {"visible_text", "visible_objects", "pointed_or_held_objects", "temporal_events", "spatial_relations", "uncertainty_notes", "evidence_frames"} else "")
        state["video_path"] = manifest.get("video_path", "")
        state["path_status"] = manifest.get("path_status", "")
        state["scenario"] = scenario
        state["evidence_frames"] = manifest.get("frames", [])
    out_json.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [f"{k}: {json.dumps(v, ensure_ascii=False)}" for k, v in state.items() if k in FIELDS]
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return state


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--scenario-number", type=int, required=True)
    parser.add_argument("--task-index", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    state = build_visual_state(args.scenario, args.scenario_number, args.task_index, args.force)
    print(json.dumps(state, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
