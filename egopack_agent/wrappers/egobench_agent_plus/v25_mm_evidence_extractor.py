#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V25-new multimodal evidence extractor.

Builds a compact evidence table from instruction, existing Qwen3-VL cards,
optional contact sheets, optional subtitles/transcripts, and DB canonical
matching.  The extractor does not inspect GT and does not answer the task.
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List

from .v25_evidence_entity_matcher import compact_db_entity_list, match_entities, norm_text


CODEX = Path(os.environ.get("CODEX_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/codex"))
EGO = Path(os.environ.get("EGO_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/EgoBench"))


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return default


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _safe_json_from_text(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?", "", s, flags=re.I).strip()
        s = re.sub(r"```$", "", s).strip()
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass
    m = re.search(r"\{.*\}", s, flags=re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}
    return {}


def _qwen_texts(qwen: Dict[str, Any]) -> Dict[str, List[str]]:
    visible_text: List[str] = []
    package_text: List[str] = []
    menu_text: List[str] = []
    for key in ("visible_text", "category_country_brand_taste_clues", "restaurant_menu_order_clues"):
        val = qwen.get(key)
        if isinstance(val, list):
            for x in val:
                if isinstance(x, dict):
                    visible_text.extend(str(v) for v in x.values() if isinstance(v, (str, int, float)))
                else:
                    visible_text.append(str(x))
        elif isinstance(val, str):
            visible_text.append(val)
    if qwen.get("scenario") in {"order", "restaurant"}:
        menu_text = visible_text[:]
    if qwen.get("scenario") == "retail":
        package_text = visible_text[:]
    return {
        "visible_text": list(dict.fromkeys([x for x in visible_text if norm_text(x)])),
        "package_text": list(dict.fromkeys([x for x in package_text if norm_text(x)])),
        "menu_text": list(dict.fromkeys([x for x in menu_text if norm_text(x)])),
    }


def _qwen_entities(qwen: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    keys = [
        ("top_k_candidates", ""),
        ("visible_products", "product"),
        ("visible_dishes", "dish"),
        ("visible_ingredients", "ingredient"),
        ("pointed_or_held_objects", "object"),
        ("relative_location_objects", "object"),
    ]
    for key, default_type in keys:
        val = qwen.get(key)
        if not isinstance(val, list):
            continue
        for item in val:
            if isinstance(item, dict):
                raw = item.get("entity") or item.get("product_name") or item.get("name") or item.get("label_text") or item.get("object")
                typ = item.get("type") or default_type or "object"
                conf = item.get("confidence", item.get("score", 0.55))
                reason = item.get("evidence") or item.get("reason") or ""
                loc = item.get("location") or item.get("relative_position") or ""
            else:
                raw, typ, conf, reason, loc = str(item), default_type or "object", 0.55, "", ""
            if norm_text(raw):
                out.append(
                    {
                        "raw_name": str(raw),
                        "canonical_db_name": "",
                        "type": str(typ),
                        "confidence": float(conf or 0.55),
                        "evidence": ["qwen3vl_card"],
                        "location": str(loc),
                        "reason": str(reason),
                    }
                )
    return out


def _iter_slot_raw_values(value: Any) -> List[str]:
    out: List[str] = []
    if isinstance(value, dict):
        for key in ("canonical_name", "canonical_db_name", "raw_name", "entity", "name", "product_name", "dish_name", "set_meal_name", "ingredient_name", "recipe_name"):
            if value.get(key):
                out.append(str(value.get(key)))
    elif isinstance(value, list):
        for item in value:
            out.extend(_iter_slot_raw_values(item))
    elif value:
        out.append(str(value))
    return [x for x in out if norm_text(x)]


def _slot_priority_terms(parsed_slots: Dict[str, Any], vision_entities: List[Dict[str, Any]], slot: str, typ: str) -> List[str]:
    terms: List[str] = []
    terms.extend(_iter_slot_raw_values(parsed_slots.get(slot)))
    type_aliases = {
        "product": {"product", "product_name", "wine", "bottle"},
        "dish": {"dish", "menu_item", "food"},
        "set_meal": {"set_meal", "meal", "set meal"},
        "restaurant": {"restaurant"},
        "ingredient": {"ingredient", "food"},
        "recipe": {"recipe", "dish"},
        "category": {"category", "section"},
    }
    aliases = type_aliases.get(typ, {typ})
    for ent in vision_entities:
        if not isinstance(ent, dict):
            continue
        et = norm_text(ent.get("type"))
        conf = float(ent.get("confidence", 0.0) or 0.0)
        if et in aliases or typ in et:
            for key in ("canonical_db_name", "raw_name", "reason"):
                if ent.get(key) and (conf >= 0.55 or key == "canonical_db_name"):
                    terms.append(str(ent.get(key)))
    return list(dict.fromkeys([x for x in terms if norm_text(x)]))


def _prioritize_matches(matches: List[Dict[str, Any]], priority_terms: List[str]) -> List[Dict[str, Any]]:
    if not matches or not priority_terms:
        return matches
    out: List[Dict[str, Any]] = []
    for item in matches:
        name = item.get("canonical_name", "")
        bonus = 0.0
        reasons = list(item.get("raw_mentions") or [])
        for raw in priority_terms:
            nr = norm_text(raw)
            nn = norm_text(name)
            if not nr or not nn:
                continue
            if nn in nr or nr in nn:
                bonus = max(bonus, 2.0)
                reasons.append(f"priority_vision_slot:{raw}")
            else:
                # Token-level bridge for labels such as "Riunite" -> "riunite moscato".
                toks = [t for t in re.findall(r"[a-z0-9']+", nr) if len(t) > 3]
                if any(t in nn for t in toks):
                    bonus = max(bonus, 1.5)
                    reasons.append(f"priority_vision_token:{raw}")
        row = dict(item)
        row["_priority_score"] = bonus
        if bonus:
            row["raw_mentions"] = list(dict.fromkeys(reasons))[:8]
            row["matched_by"] = list(dict.fromkeys(list(row.get("matched_by") or []) + ["priority_vision_slot"]))
        out.append(row)
    out.sort(key=lambda x: (float(x.get("_priority_score", 0)), float(x.get("score", 0)), len(x.get("raw_mentions") or [])), reverse=True)
    for row in out:
        row.pop("_priority_score", None)
    return out


def _find_video_path(row: Dict[str, Any]) -> str:
    for key in ("video_path", "image_path", "image_name", "video", "image"):
        val = row.get(key)
        if not val:
            continue
        p = Path(str(val))
        if p.exists():
            return str(p)
        base = p.name
        cand = EGO / "videos" / base
        if cand.exists():
            return str(cand)
        if not base.endswith(".mp4"):
            cand = EGO / "videos" / (base + ".mp4")
            if cand.exists():
                return str(cand)
    return ""


def _find_contact_sheet(spec: str, local_pos: int) -> str:
    candidates = [
        CODEX / "visual_cache_v25_new" / "contact_sheets" / f"{spec}_{local_pos + 1}.jpg",
        CODEX / "visual_cache" / f"{spec}_{local_pos + 1}" / "contact_sheet.jpg",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return ""


def _find_asr_or_subtitle(video_path: str) -> Dict[str, Any]:
    if not video_path:
        return {"transcript": "", "spoken_entities": [], "confidence": 0.0, "source": "none"}
    p = Path(video_path)
    for suffix in (".srt", ".vtt", ".txt"):
        cand = p.with_suffix(suffix)
        if cand.exists():
            text = cand.read_text(encoding="utf-8", errors="replace")
            ents = re.findall(r"[A-Z][A-Za-z0-9&' -]{2,}", text)
            return {"transcript": text[:3000], "spoken_entities": ents[:30], "confidence": 0.7, "source": suffix.lstrip(".")}
    return {"transcript": "", "spoken_entities": [], "confidence": 0.0, "source": "none"}


def _load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        if s.startswith("export "):
            s = s[len("export ") :]
        key, val = s.split("=", 1)
        val = val.strip().strip('"').strip("'")
        if key.strip() and key.strip() not in os.environ:
            os.environ[key.strip()] = val


def _image_data_url(path: str) -> str:
    if not path:
        return ""
    p = Path(path)
    if not p.exists():
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(p.read_bytes()).decode("ascii")


def _call_gpt55_vision(
    *,
    row: Dict[str, Any],
    scenario: str,
    db_summary: Dict[str, List[str]],
    qwen: Dict[str, Any],
    contact_sheet_path: str,
) -> Dict[str, Any]:
    if os.environ.get("V25_DISABLE_GPT55_VISION") == "1":
        return {"status": "skipped", "reason": "V25_DISABLE_GPT55_VISION=1"}
    _load_env(CODEX / "state" / ".openai_env")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return {"status": "skipped", "reason": "OPENAI_API_KEY missing"}
    image_url = _image_data_url(contact_sheet_path)
    if not image_url:
        return {"status": "skipped", "reason": "contact_sheet missing"}
    try:
        from openai import OpenAI
    except Exception as exc:
        return {"status": "error", "reason": f"openai import failed: {type(exc).__name__}: {exc}"}
    base_url = os.environ.get("TRACK2_OPENAI_BASE_URL") or os.environ.get("SERVICE_MODEL_API_BASE") or os.environ.get("OPENAI_BASE_URL")
    model = os.environ.get("TRACK2_OPENAI_MODEL") or os.environ.get("OPENAI_MODEL") or "gpt-5.5"
    timeout_s = float(os.environ.get("TRACK2_OPENAI_TIMEOUT", "90"))
    client_kwargs = {"api_key": api_key, "timeout": timeout_s, "max_retries": 1}
    if base_url:
        client_kwargs["base_url"] = base_url
    if os.environ.get("TRACK2_OPENAI_NO_PROXY", "1") != "0":
        try:
            import httpx

            client_kwargs["http_client"] = httpx.Client(timeout=timeout_s, trust_env=False)
        except Exception:
            pass
    client = OpenAI(**client_kwargs)
    prompt = {
        "task": "Extract multimodal evidence only. Do not solve the task and do not produce tool calls.",
        "scenario": scenario,
        "utterance": row.get("Instruction", ""),
        "qwen3vl_card_summary": {
            "status": qwen.get("status"),
            "scene_summary": qwen.get("scene_summary"),
            "visible_text": qwen.get("visible_text", [])[:30],
            "top_k_candidates": qwen.get("top_k_candidates", [])[:8],
        },
        "db_canonical_names": db_summary,
        "required_json_schema": {
            "key_regions": [{"region_id": "string", "description": "string", "visible_text": [], "relative_position": "string"}],
            "ocr_visible_text": [],
            "vision_entities": [{"raw_name": "string", "canonical_db_name": "string", "type": "product|dish|set_meal|restaurant|ingredient|recipe|object", "confidence": 0.0, "evidence": [], "location": "string", "reason": "string"}],
            "candidate_slots": {"primary_product": [], "dish": [], "set_meal": [], "ingredient": [], "recipe": [], "restaurant": [], "branch_attributes": [], "mutation_intent": "string", "closure_needed": []},
            "uncertainty": {"visual_grounding_failed": False, "entity_ambiguous": False, "needs_tool_query": True, "notes": "string"},
        },
    }
    started = time.time()
    try:
        resp = client.responses.create(
            model=model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": json.dumps(prompt, ensure_ascii=False)},
                        {"type": "input_image", "image_url": image_url},
                    ],
                }
            ],
            max_output_tokens=1800,
        )
        text = getattr(resp, "output_text", "") or ""
        return {"status": "success", "api": "responses", "model": model, "latency": round(time.time() - started, 3), "raw_text": text, "parsed": _safe_json_from_text(text)}
    except Exception as exc:
        responses_error = f"{type(exc).__name__}: {exc}"
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": json.dumps(prompt, ensure_ascii=False)},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
            max_tokens=1800,
            temperature=0,
        )
        text = resp.choices[0].message.content or ""
        return {"status": "success", "api": "chat_completions", "model": model, "latency": round(time.time() - started, 3), "raw_text": text, "parsed": _safe_json_from_text(text), "responses_error": responses_error}
    except Exception as exc:
        return {"status": "error", "model": model, "latency": round(time.time() - started, 3), "reason": f"responses={responses_error}; chat={type(exc).__name__}: {exc}"}


def build_evidence_table(
    *,
    row: Dict[str, Any],
    scenario: str,
    spec: str,
    local_pos: int,
    db: Any,
    qwen_card: Dict[str, Any],
    use_gpt_vision: bool = True,
) -> Dict[str, Any]:
    video_path = _find_video_path(row)
    contact_sheet = _find_contact_sheet(spec, local_pos)
    qtext = _qwen_texts(qwen_card)
    qents = _qwen_entities(qwen_card)
    asr = _find_asr_or_subtitle(video_path)
    db_summary = compact_db_entity_list(scenario, db, limit_per_type=70)
    gpt = _call_gpt55_vision(
        row=row,
        scenario=scenario,
        db_summary=db_summary,
        qwen=qwen_card,
        contact_sheet_path=contact_sheet,
    ) if use_gpt_vision else {"status": "skipped", "reason": "driver_disabled"}
    parsed = gpt.get("parsed") if isinstance(gpt, dict) else {}
    if not isinstance(parsed, dict):
        parsed = {}
    key_regions = parsed.get("key_regions") if isinstance(parsed.get("key_regions"), list) else []
    ocr_visible = list(qtext["visible_text"])
    for x in parsed.get("ocr_visible_text") or []:
        if isinstance(x, (str, int, float)):
            ocr_visible.append(str(x))
    vision_entities = list(qents)
    for ent in parsed.get("vision_entities") or []:
        if isinstance(ent, dict):
            vision_entities.append(
                {
                    "raw_name": ent.get("raw_name") or ent.get("canonical_db_name") or "",
                    "canonical_db_name": ent.get("canonical_db_name") or "",
                    "type": ent.get("type") or "object",
                    "confidence": float(ent.get("confidence", 0.6) or 0.6),
                    "evidence": ent.get("evidence") or ["gpt55_vision"],
                    "location": ent.get("location") or "",
                    "reason": ent.get("reason") or "",
                }
            )
    evidence = {
        "task_key": f"{spec}::{local_pos}",
        "spec": spec,
        "index": int(row.get("_v8_original_index", local_pos)),
        "local_pos": local_pos,
        "scenario": scenario,
        "utterance": row.get("Instruction", ""),
        "frame_evidence": {
            "sampled_frames": [],
            "contact_sheet_path": contact_sheet,
            "video_path": video_path,
            "key_regions": key_regions,
        },
        "ocr_evidence": {
            "visible_text": list(dict.fromkeys([x for x in ocr_visible if norm_text(x)])),
            "menu_text": qtext["menu_text"],
            "package_text": qtext["package_text"],
            "price_text": [x for x in ocr_visible if re.search(r"\d+[.]\d+|\d+ yuan|rmb", str(x), re.I)],
            "confidence": 0.75 if ocr_visible else 0.0,
        },
        "asr_evidence": asr,
        "vision_entities": vision_entities,
        "candidate_slots": parsed.get("candidate_slots") if isinstance(parsed.get("candidate_slots"), dict) else {},
        "uncertainty": {
            "visual_grounding_failed": not bool(vision_entities),
            "entity_ambiguous": False,
            "needs_tool_query": True,
            "notes": "",
        },
        "sources": {
            "qwen_card_path": qwen_card.get("_path", ""),
            "qwen_status": qwen_card.get("status"),
            "gpt55_vision_status": gpt.get("status") if isinstance(gpt, dict) else "not_called",
            "gpt55_vision_reason": gpt.get("reason", "") if isinstance(gpt, dict) else "",
            "gpt55_vision_model": gpt.get("model", "") if isinstance(gpt, dict) else "",
            "gpt55_vision_latency": gpt.get("latency", 0) if isinstance(gpt, dict) else 0,
        },
        "final_run": False,
        "uses_final_hidden_metadata": False,
        "uses_val41_gt_for_runtime": False,
    }
    matches = match_entities(scenario, db, evidence, top_k=5)
    evidence["canonical_matches"] = matches
    parsed_slots = dict(evidence.get("candidate_slots") or {})
    slots = dict(parsed_slots)
    map_keys = {
        "product": "primary_product",
        "dish": "dish",
        "set_meal": "set_meal",
        "restaurant": "restaurant",
        "ingredient": "ingredient",
        "recipe": "recipe",
        "category": "category",
    }
    for typ, slot in map_keys.items():
        ranked_matches = _prioritize_matches(matches.get(typ, []), _slot_priority_terms(parsed_slots, vision_entities, slot, typ))
        vals = [
            {
                "canonical_name": x.get("canonical_name"),
                "score": x.get("score"),
                "risk": x.get("risk"),
                "matched_by": x.get("matched_by"),
                "raw_mentions": x.get("raw_mentions"),
                "meta": x.get("meta"),
            }
            for x in ranked_matches
            if float(x.get("score", 0) or 0) >= 0.42
        ]
        if vals:
            slots[slot] = vals
    evidence["candidate_slots"] = slots
    evidence["uncertainty"]["entity_ambiguous"] = any(len(v) > 1 for v in slots.values() if isinstance(v, list))
    evidence["uncertainty"]["visual_grounding_failed"] = not any(slots.get(k) for k in ("primary_product", "dish", "set_meal", "ingredient", "recipe"))
    return evidence


def save_evidence(cache_path: Path, evidence: Dict[str, Any]) -> None:
    _write_json(cache_path, evidence)
