#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.request
from pathlib import Path
from typing import Any

CODEX = Path("/home/data-gxu/acm/egolink2026-main/code/track2/codex")

LIST_FIELDS = [
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
]

PROTECTED_KEYS = [
    "teacher",
    "error",
    "scenario",
    "scenario_spec",
    "task_id",
    "cache_key",
    "video_path",
    "path_status",
    "source_split",
    "source_file",
    "instruction_digest",
    "final_hidden_metadata_used",
    "_qwen3vl_json_parse_status",
]


def normalize_list(value: Any) -> list[Any]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return [value]
    text = str(value).strip()
    if not text:
        return []
    parts = [p.strip(" -;\t") for p in re.split(r"[\n;]+", text) if p.strip(" -;\t")]
    return parts or [text]


def normalize_topk(value: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in normalize_list(value):
        if isinstance(item, dict):
            entity = item.get("entity") or item.get("name") or item.get("text") or item.get("label") or ""
            typ = item.get("type") or item.get("entity_type") or "unknown"
            evidence = item.get("evidence") or item.get("reason") or item.get("description") or ""
            conf = item.get("confidence", 0.0)
        else:
            entity = str(item)
            typ = "unknown"
            evidence = "raw top_k item"
            conf = 0.2
        entity = " ".join(str(entity).strip().split())
        if not entity or entity.lower() in {"unknown", "none", "n/a", "null"}:
            continue
        try:
            conf_f = float(conf)
        except Exception:
            conf_f = 0.0
        out.append(
            {
                "entity": entity[:160],
                "type": str(typ or "unknown")[:80],
                "evidence": str(evidence or "")[:320],
                "confidence": max(0.0, min(1.0, conf_f)),
            }
        )
    dedup: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in out:
        key = (item["entity"].lower(), item["type"].lower())
        if key in seen:
            continue
        seen.add(key)
        dedup.append(item)
    return dedup[:10]


def load_env() -> None:
    env_file = CODEX / "state/.openai_env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("export ") or "=" not in line:
            continue
        key, value = line[len("export ") :].split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def gpt55_clean(card: dict[str, Any], timeout: int = 180) -> dict[str, Any] | None:
    load_env()
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("SERVICE_MODEL_API_KEY")
    if not key:
        return None
    base = (os.environ.get("TRACK2_OPENAI_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or "https://ai-pixel.online/v1").rstrip("/")
    model = os.environ.get("TRACK2_OPENAI_MODEL") or "gpt-5.5"
    visual_only = {
        key_name: card.get(key_name)
        for key_name in [
            "status",
            "teacher",
            "scenario",
            "scenario_spec",
            "task_id",
            "cache_key",
            "video_path",
            "path_status",
            "scene_summary",
            "visible_text",
            "readable_labels_text",
            "visible_products",
            "visible_dishes",
            "visible_ingredients",
            "pointed_or_held_objects",
            "relative_location_objects",
            "category_country_brand_taste_clues",
            "restaurant_menu_order_clues",
            "top_k_candidates",
            "uncertainty_notes",
            "raw_output_preview",
            "_qwen3vl_json_parse_status",
        ]
    }
    prompt = (
        "Clean this Qwen3-VL visual grounding card into strict JSON. "
        "Use only visual evidence present in the card fields and raw_output_preview. "
        "Do not use task instruction text as evidence. Do not infer hidden answers. "
        "If visual entities can be inferred, fill top_k_candidates with candidate objects and low confidence when uncertain. "
        "If no entity can be visually inferred, return empty top_k_candidates and explain uncertainty. "
        "Return only a JSON object with keys: scene_summary, visible_text, visible_products, visible_dishes, visible_ingredients, "
        "pointed_or_held_objects, relative_location_objects, category_country_brand_taste_clues, restaurant_menu_order_clues, "
        "top_k_candidates, uncertainty_notes.\n\n"
        + json.dumps(visual_only, ensure_ascii=False)[:22000]
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You output strict JSON only."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": 1800,
    }
    req = urllib.request.Request(
        base + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        content = data["choices"][0]["message"]["content"].strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.S)
        start, end = content.find("{"), content.rfind("}")
        if start >= 0 and end > start:
            content = content[start : end + 1]
        cleaned = json.loads(content)
        cleaned["_gpt55_cleaned"] = True
        cleaned["_gpt55_model"] = model
        return cleaned
    except Exception as exc:
        return {"_gpt55_clean_error": f"{type(exc).__name__}:{str(exc)[:240]}"}


def needs_gpt_clean(card: dict[str, Any], force_all: bool) -> bool:
    if force_all:
        return True
    if not card.get("top_k_candidates"):
        return True
    if card.get("_qwen3vl_json_parse_status") in {"malformed_json_raw_scene", "empty"}:
        return True
    return False


def clean_card(card: dict[str, Any], use_gpt55: bool, force_all: bool) -> dict[str, Any]:
    for field in LIST_FIELDS:
        card[field] = normalize_list(card.get(field))
    card["top_k_candidates"] = normalize_topk(card.get("top_k_candidates"))
    if use_gpt55 and needs_gpt_clean(card, force_all):
        cleaned = gpt55_clean(card)
        if cleaned and "_gpt55_clean_error" not in cleaned:
            protected = {key: card.get(key) for key in PROTECTED_KEYS if key in card}
            status_before = card.get("status")
            card.update(cleaned)
            card.update(protected)
            if status_before == "qwen3vl_success":
                card["status"] = "qwen3vl_success"
            for field in LIST_FIELDS:
                card[field] = normalize_list(card.get(field))
            card["top_k_candidates"] = normalize_topk(card.get("top_k_candidates"))
        elif cleaned and "_gpt55_clean_error" in cleaned:
            card["_gpt55_clean_error"] = cleaned["_gpt55_clean_error"]
    if not card.get("top_k_candidates"):
        card["status"] = "grounding_failed"
        card["teacher"] = "qwen3vl"
        card.setdefault("uncertainty_notes", [])
        if not isinstance(card["uncertainty_notes"], list):
            card["uncertainty_notes"] = normalize_list(card["uncertainty_notes"])
        note = "grounding_failed: no top_k_candidates after GPT-5.5 structured cleaning."
        if note not in card["uncertainty_notes"]:
            card["uncertainty_notes"].append(note)
    card["final_hidden_metadata_used"] = False
    return card


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--card_dir", required=True)
    parser.add_argument("--use_gpt55", action="store_true")
    parser.add_argument("--force_all", action="store_true")
    args = parser.parse_args()

    card_dir = Path(args.card_dir)
    summary: list[dict[str, Any]] = []
    for path in sorted(p for p in card_dir.glob("*.json") if not p.name.startswith("manifest") and "summary" not in p.name):
        try:
            card = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            card = {
                "status": "grounding_failed",
                "teacher": "qwen3vl",
                "cache_key": path.stem,
                "top_k_candidates": [],
                "uncertainty_notes": [f"json_parse_failed:{type(exc).__name__}"],
                "final_hidden_metadata_used": False,
            }
        card = clean_card(card, use_gpt55=args.use_gpt55, force_all=args.force_all)
        card["card_cleaned_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        path.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
        row = {
            "file": path.name,
            "status": card.get("status"),
            "teacher": card.get("teacher"),
            "parse_status": card.get("_qwen3vl_json_parse_status"),
            "top_k_count": len(card.get("top_k_candidates") or []),
            "gpt55_cleaned": bool(card.get("_gpt55_cleaned")),
            "gpt55_error": card.get("_gpt55_clean_error", ""),
        }
        summary.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    out = card_dir / f"strict_cleaning_summary_all_dev_offline_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps({"items": summary}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("summary", out, flush=True)


if __name__ == "__main__":
    main()
