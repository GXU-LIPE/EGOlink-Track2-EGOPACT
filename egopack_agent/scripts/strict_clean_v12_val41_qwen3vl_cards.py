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
    return [p.strip(" -;\t") for p in re.split(r"[\n;]+", text) if p.strip(" -;\t")] or [text]


def normalize_topk(value: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in normalize_list(value):
        if isinstance(item, dict):
            entity = item.get("entity") or item.get("name") or item.get("text") or item.get("label") or ""
            typ = item.get("type") or item.get("entity_type") or "unknown"
            evidence = item.get("evidence") or item.get("reason") or item.get("description") or ""
            conf = item.get("confidence", 0.0)
        else:
            entity, typ, evidence, conf = str(item), "unknown", "raw top_k item", 0.2
        entity = " ".join(str(entity).strip().split())
        if not entity or entity.lower() in {"unknown", "none", "n/a"}:
            continue
        try:
            conf_f = float(conf)
        except Exception:
            conf_f = 0.0
        out.append({"entity": entity[:140], "type": str(typ)[:80], "evidence": str(evidence)[:260], "confidence": max(0.0, min(1.0, conf_f))})
    dedup = []
    seen = set()
    for item in out:
        key = (item["entity"].lower(), item["type"].lower())
        if key in seen:
            continue
        seen.add(key)
        dedup.append(item)
    return dedup[:10]


def load_env() -> None:
    p = CODEX / "state/.openai_env"
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("export ") or "=" not in line:
            continue
        k, v = line[len("export ") :].split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip("\"").strip("'"))


def gpt55_clean(card: dict[str, Any], timeout: int = 180) -> dict[str, Any] | None:
    load_env()
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("SERVICE_MODEL_API_KEY")
    if not key:
        return None
    base = (os.environ.get("TRACK2_OPENAI_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or "https://ai-pixel.online/v1").rstrip("/")
    model = os.environ.get("TRACK2_OPENAI_MODEL") or "gpt-5.5"
    visual_only = {k: card.get(k) for k in [
        "status",
        "teacher",
        "scenario",
        "scenario_spec",
        "task_id",
        "original_index",
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
    ]}
    prompt = (
        "Clean this Qwen3-VL visual grounding card into strict JSON. "
        "Use only the visual evidence present in the card, especially scene_summary/raw_output_preview/visible fields. "
        "Do not use task instruction text as evidence and do not invent hidden answers. "
        "If entities can be inferred, fill top_k_candidates with candidate objects. "
        "If no entity can be visually inferred, return an empty top_k_candidates list and explain uncertainty. "
        "Return only a JSON object with: scene_summary, visible_text, visible_products, visible_dishes, visible_ingredients, "
        "pointed_or_held_objects, relative_location_objects, category_country_brand_taste_clues, restaurant_menu_order_clues, top_k_candidates, uncertainty_notes.\n\n"
        + json.dumps(visual_only, ensure_ascii=False)[:14000]
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You output strict JSON only."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": 1600,
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
    except Exception:
        return None


def clean_card(card: dict[str, Any], use_gpt55: bool) -> dict[str, Any]:
    protected_keys = [
        "status",
        "teacher",
        "error",
        "scenario",
        "scenario_spec",
        "task_id",
        "original_index",
        "cache_key",
        "video_path",
        "path_status",
        "source_split",
        "final_hidden_metadata_used",
    ]
    for field in LIST_FIELDS:
        card[field] = normalize_list(card.get(field))
    card["top_k_candidates"] = normalize_topk(card.get("top_k_candidates"))
    if use_gpt55 and not card["top_k_candidates"]:
        cleaned = gpt55_clean(card)
        if cleaned:
            protected = {k: card.get(k) for k in protected_keys if k in card}
            card.update(cleaned)
            card.update(protected)
            for field in LIST_FIELDS:
                card[field] = normalize_list(card.get(field))
            card["top_k_candidates"] = normalize_topk(card.get("top_k_candidates"))
    if not card["top_k_candidates"]:
        card["status"] = "grounding_failed"
        card.setdefault("uncertainty_notes", [])
        if isinstance(card["uncertainty_notes"], list):
            card["uncertainty_notes"].append("grounding_failed: no top_k_candidates after GPT-5.5 structured cleaning.")
    return card


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--card_dir", default=str(CODEX / "visual_cache_v12/qwen3vl_grounding"))
    ap.add_argument("--use_gpt55", action="store_true")
    args = ap.parse_args()
    card_dir = Path(args.card_dir)
    summary = []
    for path in sorted(p for p in card_dir.glob("*.json") if not p.name.startswith("manifest") and not p.name.startswith("cleaning_summary")):
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
        card = clean_card(card, use_gpt55=args.use_gpt55)
        card["card_cleaned_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        path.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
        row = {
            "file": path.name,
            "status": card.get("status"),
            "teacher": card.get("teacher"),
            "top_k_count": len(card.get("top_k_candidates") or []),
            "gpt55_cleaned": bool(card.get("_gpt55_cleaned")),
        }
        summary.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    out = card_dir / f"strict_cleaning_summary_val41_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps({"items": summary}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("summary", out)


if __name__ == "__main__":
    main()
