#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

CODEX = Path("/home/data-gxu/acm/egolink2026-main/code/track2/codex")


EXPECTED_LIST_FIELDS = [
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
    parts = [p.strip(" -;\t") for p in re.split(r"[\n;]+", text) if p.strip(" -;\t")]
    return parts if parts else [text]


def text_blob(card: dict[str, Any]) -> str:
    chunks = []
    for key in [
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
        "raw_output_preview",
        "instruction_digest",
    ]:
        value = card.get(key)
        if isinstance(value, (list, tuple)):
            chunks.extend(str(x) for x in value)
        elif isinstance(value, dict):
            chunks.append(json.dumps(value, ensure_ascii=False))
        elif value:
            chunks.append(str(value))
    return "\n".join(chunks)


def heuristic_candidates(card: dict[str, Any]) -> list[dict[str, Any]]:
    scenario = str(card.get("scenario", "")).lower()
    blob = text_blob(card)
    candidates: list[dict[str, Any]] = []

    def add(entity: str, typ: str, evidence: str, confidence: float = 0.35) -> None:
        entity = " ".join(str(entity).strip().strip("\"'`.,;:()[]{}").split())
        if len(entity) < 2:
            return
        low = entity.lower()
        if low in {"unknown", "none", "n/a", "visible", "item", "object", "product", "dish"}:
            return
        for c in candidates:
            if c["entity"].lower() == low and c["type"] == typ:
                return
        candidates.append({"entity": entity[:120], "type": typ, "evidence": evidence[:200], "confidence": confidence})

    typed_fields = [
        ("visible_products", "product"),
        ("visible_dishes", "dish"),
        ("visible_ingredients", "ingredient"),
        ("restaurant_menu_order_clues", "dish" if scenario in {"order", "restaurant"} else "restaurant"),
        ("category_country_brand_taste_clues", "category"),
        ("visible_text", "product" if scenario == "retail" else "dish"),
        ("readable_labels_text", "product" if scenario == "retail" else "dish"),
    ]
    for field, typ in typed_fields:
        for item in normalize_list(card.get(field)):
            if isinstance(item, dict):
                entity = item.get("entity") or item.get("name") or item.get("text") or item.get("label") or ""
                evidence = item.get("evidence") or json.dumps(item, ensure_ascii=False)
            else:
                entity = str(item)
                evidence = f"from {field}"
            add(entity, typ, evidence, 0.45)

    # Pull obvious quoted or capitalized menu/product names from raw text.
    for m in re.finditer(r"[\"“]([^\"”]{3,80})[\"”]", blob):
        typ = "product" if scenario == "retail" else "dish"
        add(m.group(1), typ, "quoted visual text", 0.35)
    for m in re.finditer(r"\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z0-9&'-]+){1,5})\b", blob):
        ent = m.group(1)
        if ent.startswith(("The ", "This ", "There ", "Visible ", "Task ", "Scenario ")):
            continue
        typ = "product" if scenario == "retail" else ("ingredient" if scenario == "kitchen" else "dish")
        add(ent, typ, "capitalized visual text", 0.25)
    return candidates[:8]


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("export ") or "=" not in line:
            continue
        k, v = line[len("export ") :].split("=", 1)
        v = v.strip().strip("\"").strip("'")
        os.environ.setdefault(k.strip(), v)


def gpt55_clean(card: dict[str, Any], timeout: int = 180) -> dict[str, Any] | None:
    load_env_file(CODEX / "state/.openai_env")
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("SERVICE_MODEL_API_KEY")
    base = os.environ.get("TRACK2_OPENAI_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or "https://ai-pixel.online/v1"
    model = os.environ.get("TRACK2_OPENAI_MODEL") or "gpt-5.5"
    if not key:
        return None
    base = base.rstrip("/")
    prompt = {
        "role": "user",
        "content": (
            "Clean this Qwen3-VL visual grounding card into strict JSON. "
            "Do not invent hidden answers. Preserve uncertainty. "
            "Always provide top_k_candidates if any entity can be inferred from visual text, labels, objects, menu items, products, dishes, or ingredients. "
            "Return only a JSON object with fields: scene_summary, visible_text, visible_products, visible_dishes, visible_ingredients, "
            "pointed_or_held_objects, relative_location_objects, category_country_brand_taste_clues, restaurant_menu_order_clues, "
            "top_k_candidates, uncertainty_notes.\n\n"
            + json.dumps(card, ensure_ascii=False)[:12000]
        ),
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You output strict JSON only."},
            prompt,
        ],
        "temperature": 0,
        "max_tokens": 1400,
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
        start = content.find("{")
        end = content.rfind("}")
        if start >= 0 and end > start:
            content = content[start : end + 1]
        cleaned = json.loads(content)
        cleaned["_gpt55_cleaned"] = True
        cleaned["_gpt55_model"] = model
        return cleaned
    except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError, OSError):
        return None


def normalize_card(card: dict[str, Any], use_gpt55: bool) -> dict[str, Any]:
    for field in EXPECTED_LIST_FIELDS:
        card[field] = normalize_list(card.get(field))
    topk = normalize_list(card.get("top_k_candidates"))
    normalized_topk = []
    for item in topk:
        if isinstance(item, dict):
            ent = item.get("entity") or item.get("name") or item.get("text") or item.get("label") or ""
            typ = item.get("type") or item.get("entity_type") or "unknown"
            evidence = item.get("evidence") or item.get("reason") or item.get("description") or ""
            conf = item.get("confidence", 0.0)
            try:
                conf = float(conf)
            except Exception:
                conf = 0.0
            if str(ent).strip():
                normalized_topk.append(
                    {
                        "entity": str(ent).strip(),
                        "type": str(typ).strip() or "unknown",
                        "evidence": str(evidence).strip(),
                        "confidence": max(0.0, min(1.0, conf)),
                    }
                )
        elif str(item).strip():
            normalized_topk.append({"entity": str(item).strip(), "type": "unknown", "evidence": "raw top_k item", "confidence": 0.2})
    card["top_k_candidates"] = normalized_topk[:8]

    if not card["top_k_candidates"]:
        heur = heuristic_candidates(card)
        if heur:
            card["top_k_candidates"] = heur
            card.setdefault("uncertainty_notes", []).append("top_k_candidates filled by deterministic card cleaner from visible fields.")
            card["_heuristic_cleaned"] = True

    if use_gpt55 and not card["top_k_candidates"]:
        cleaned = gpt55_clean(card)
        if cleaned:
            protected = {k: card.get(k) for k in ["status", "teacher", "scenario", "scenario_spec", "task_id", "original_index", "cache_key", "video_path", "path_status", "source_split", "final_hidden_metadata_used"]}
            for k, v in cleaned.items():
                card[k] = v
            card.update({k: v for k, v in protected.items() if v is not None})
            for field in EXPECTED_LIST_FIELDS:
                card[field] = normalize_list(card.get(field))
            card = normalize_card(card, use_gpt55=False)

    if not card.get("top_k_candidates"):
        card["status"] = "grounding_failed"
        card["teacher"] = card.get("teacher") or "qwen3vl"
        card.setdefault("uncertainty_notes", []).append("grounding_failed: no top_k_candidates after deterministic and GPT-5.5 cleaning.")
    elif card.get("status") == "grounding_failed":
        card["status"] = "qwen3vl_cleaned"
    return card


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--card_dir", default=str(CODEX / "visual_cache_v12/qwen3vl_grounding"))
    parser.add_argument("--use_gpt55", action="store_true")
    args = parser.parse_args()
    card_dir = Path(args.card_dir)
    files = sorted(p for p in card_dir.glob("*.json") if not p.name.startswith("manifest"))
    summary = []
    for path in files:
        try:
            card = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            card = {
                "status": "grounding_failed",
                "teacher": "qwen3vl",
                "cache_key": path.stem,
                "scene_summary": "",
                "top_k_candidates": [],
                "uncertainty_notes": [f"json_parse_failed:{type(exc).__name__}"],
                "final_hidden_metadata_used": False,
            }
        card = normalize_card(card, use_gpt55=args.use_gpt55)
        card["card_cleaned_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        path.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
        summary.append(
            {
                "file": path.name,
                "status": card.get("status"),
                "teacher": card.get("teacher"),
                "task_id": card.get("task_id"),
                "original_index": card.get("original_index"),
                "top_k_count": len(card.get("top_k_candidates") or []),
                "gpt55_cleaned": bool(card.get("_gpt55_cleaned")),
                "heuristic_cleaned": bool(card.get("_heuristic_cleaned")),
            }
        )
        print(json.dumps(summary[-1], ensure_ascii=False), flush=True)
    out = card_dir / f"cleaning_summary_val41_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps({"items": summary}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("summary", out, flush=True)


if __name__ == "__main__":
    main()
