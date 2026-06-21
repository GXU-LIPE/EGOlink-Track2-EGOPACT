#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V24 target-task visual/entity second pass.

This script attempts a compact OpenAI-compatible API cleanup if credentials are
available, but always writes deterministic fallback JSON.  It never uses GT.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List


CODEX = Path(os.environ.get("CODEX_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/codex"))
EGO = Path(os.environ.get("EGO_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/EgoBench"))
SPLIT_DIR = CODEX / "state" / "materialized_splits" / "validation_A_limit30"


def norm_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return default


def append_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def qwen_card(spec: str, pos: int) -> Dict[str, Any]:
    for p in [
        CODEX / "visual_cache_v12" / "qwen3vl_grounding" / f"{spec}_{pos+1}.json",
        CODEX / "visual_cache_v12" / "backups" / "qwen3vl_grounding_before_val41_allcards_20260619_144227" / f"{spec}_{pos+1}.json",
    ]:
        data = read_json(p)
        if isinstance(data, dict):
            data["_path"] = str(p)
            return data
    return {"status": "missing", "_path": ""}


def compact_entities(row: Dict[str, Any], qwen: Dict[str, Any], scenario: str) -> Dict[str, Any]:
    values = row.get("value") or []
    if isinstance(values, str):
        values = [values]
    candidates = []
    for v in values:
        if str(v).strip():
            candidates.append({"name": str(v), "type": "unknown", "confidence": 0.82, "evidence": "scenario current value field"})
    for obj in qwen.get("top_k_candidates") or []:
        if isinstance(obj, dict):
            name = obj.get("name") or obj.get("entity") or obj.get("product_name") or obj.get("dish_name") or obj.get("ingredient_name") or obj.get("recipe_name")
            if name:
                candidates.append({"name": name, "type": obj.get("type") or scenario, "confidence": float(obj.get("confidence", obj.get("score", 0.7)) or 0.7), "evidence": obj.get("evidence", "qwen top_k")})
    seen = set()
    dedup = []
    for c in candidates:
        key = norm_text(c["name"]) + "|" + norm_text(c.get("type"))
        if key not in seen:
            seen.add(key)
            dedup.append(c)
    instr = row.get("Instruction", "")
    attr = []
    for token in ["lowest", "highest", "cheapest", "low sugar", "low fat", "discount", "tax", "nutrition", "price", "taste", "allergen", "category", "country"]:
        if token in norm_text(instr):
            attr.append(token)
    intent = "mutation" if any(x in norm_text(instr) for x in ["add", "remove", "update", "cart", "order", "menu", "shopping list"]) else "query"
    rel = " ".join([x for x in ["left", "right", "top", "bottom", "first", "second", "third", "pointing", "shelf", "menu"] if x in norm_text(instr)])
    return {
        "primary_entities": dedup[:10],
        "relative_position": rel,
        "attribute_conditions": attr,
        "mutation_intent": intent,
        "uncertainty": "high" if not dedup else "medium" if qwen.get("status") == "grounding_failed" else "low",
        "source": "deterministic_qwen_value_fallback",
    }


def call_api(prompt: str) -> Dict[str, Any] | None:
    key = os.environ.get("OPENAI_API_KEY")
    base = (os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE") or "https://ai-pixel.online/v1").rstrip("/")
    model = os.environ.get("TRACK2_OPENAI_MODEL") or os.environ.get("OPENAI_MODEL") or "gpt-5.5"
    if not key:
        return None
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return strict JSON only. Do not use hidden ground truth."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": 700,
    }
    req = urllib.request.Request(
        base + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            obj = json.loads(resp.read().decode("utf-8", errors="replace"))
        content = obj["choices"][0]["message"]["content"]
        m = re.search(r"\{.*\}", content, flags=re.S)
        if not m:
            return None
        return json.loads(m.group(0))
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default=str(CODEX / "analysis" / "v24_target_tasks.json"))
    ap.add_argument("--output", default=str(CODEX / "analysis" / "v24_visual_second_pass.jsonl"))
    ap.add_argument("--use-api", action="store_true")
    args = ap.parse_args()
    target_obj = read_json(Path(args.targets), {})
    out = []
    for t in target_obj.get("targets", []):
        spec = t["spec"]
        pos = int(t["local_pos"])
        scenario = t["scenario"]
        rows = read_json(SPLIT_DIR / f"{spec}.json", [])
        row = rows[pos]
        qwen = qwen_card(spec, pos)
        fallback = compact_entities(row, qwen, scenario)
        result = None
        api_used = False
        if args.use_api:
            prompt = json.dumps({
                "task": "Extract visual/entity candidates for an EgoBench Track2 task. Return JSON with primary_entities, relative_position, attribute_conditions, mutation_intent, uncertainty.",
                "scenario": scenario,
                "instruction": row.get("Instruction"),
                "image_description": row.get("image_description"),
                "qwen_card": {k: qwen.get(k) for k in ["status", "scene_summary", "visible_text", "top_k_candidates", "uncertainty_notes"]},
                "current_value": row.get("value"),
            }, ensure_ascii=False)[:9000]
            result = call_api(prompt)
            api_used = bool(result)
        if not isinstance(result, dict):
            result = fallback
        result.update({"spec": spec, "index": t["index"], "local_pos": pos, "scenario": scenario, "api_used": api_used, "gt_used": False})
        out.append(result)
    append_jsonl(Path(args.output), out)
    print(json.dumps({"rows": len(out), "api_used_count": sum(1 for r in out if r.get("api_used")), "output": args.output}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

