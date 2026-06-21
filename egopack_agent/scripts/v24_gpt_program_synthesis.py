#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPT-5.5 tool-program synthesis for V24 target tasks.

Runtime inputs intentionally exclude val41 GT and analysis fields.  The model
receives instruction, Qwen visual card, tool signatures, current DB/catalog
state, and current value field only.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
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


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def as_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if hasattr(obj, "__dataclass_fields__"):
        out = asdict(obj)
    elif isinstance(obj, dict):
        out = dict(obj)
    else:
        out = dict(getattr(obj, "__dict__", {}))
    for k, v in list(out.items()):
        if hasattr(v, "__dataclass_fields__"):
            out[k] = asdict(v)
    return out


def init_db(scenario: str, number: int) -> Any:
    sys.path.insert(0, str(EGO))
    if scenario == "retail":
        from tools.retail.retail_db import RetailDB
        from tools.retail import retail_init
        db = RetailDB()
        db.init_from_json(getattr(retail_init, f"retail_init_data{number}"))
        return db
    if scenario == "restaurant":
        from tools.restaurant.restaurant_db import RestaurantDB
        from tools.restaurant import restaurant_init
        db = RestaurantDB()
        data = getattr(restaurant_init, f"restaurant_init_data{number}", None) or getattr(restaurant_init, "restaurant_init_data")
        db.init_from_json(data)
        return db
    if scenario == "order":
        from tools.order.order_db import OrderDB
        from tools.order import order_init
        db = OrderDB()
        data = getattr(order_init, f"order_init_data{number}", None) or getattr(order_init, "order_init_data")
        db.init_from_json(data)
        return db
    if scenario == "kitchen":
        from tools.kitchen.kitchen_db import KitchenDB
        from tools.kitchen import kitchen_init
        db = KitchenDB()
        data = getattr(kitchen_init, f"kitchen_init_data{number}", None) or getattr(kitchen_init, "kitchen_init_data")
        db.init_from_json(data)
        return db
    raise ValueError(scenario)


def qwen_card(spec: str, pos: int) -> Dict[str, Any]:
    for p in [
        CODEX / "visual_cache_v12" / "qwen3vl_grounding" / f"{spec}_{pos+1}.json",
        CODEX / "visual_cache_v12" / "backups" / "qwen3vl_grounding_before_val41_allcards_20260619_144227" / f"{spec}_{pos+1}.json",
    ]:
        data = read_json(p)
        if isinstance(data, dict):
            return {k: data.get(k) for k in ["status", "scene_summary", "visible_text", "top_k_candidates", "restaurant_menu_order_clues", "category_country_brand_taste_clues", "uncertainty_notes"]}
    return {"status": "missing"}


def tool_signatures(db: Any) -> List[str]:
    sigs = []
    for name in dir(db):
        if name.startswith("_"):
            continue
        fn = getattr(db, name)
        if callable(fn):
            try:
                sigs.append(f"{name}{inspect.signature(fn)}")
            except Exception:
                pass
    return sorted(sigs)


def nutrition_dict(n: Any) -> Dict[str, Any]:
    return as_dict(n)


def compact_db(scenario: str, db: Any) -> Dict[str, Any]:
    if scenario == "retail":
        products = []
        for obj in getattr(db, "catalog", {}).values():
            r = as_dict(obj)
            products.append({k: r.get(k) for k in ["name", "category", "price", "tax_rate", "discount", "nutritional_characteristics", "taste", "country_of_origin", "nutrition"]})
        return {"products": products[:200], "user_carts": str(getattr(db, "user_carts", {}))[:4000]}
    if scenario == "restaurant":
        dishes = []
        for obj in getattr(db, "catalog", {}).values():
            r = as_dict(obj)
            dishes.append({k: r.get(k) for k in ["name", "category", "price", "tax_rate", "discount", "nutritional_characteristics", "taste", "allergens", "nutrition"]})
        meals = [as_dict(x) for x in getattr(db, "set_meals", {}).values()]
        return {"dishes": dishes[:220], "set_meals": meals[:60], "user_orders": str(getattr(db, "user_orders", {}))[:5000]}
    if scenario == "order":
        restaurants = {}
        for rname, store in getattr(db, "restaurants", {}).items():
            dishes = []
            for obj in store.get("catalog", {}).values():
                r = as_dict(obj)
                dishes.append({k: r.get(k) for k in ["name", "category", "price", "tax_rate", "discount", "nutritional_characteristics", "taste", "allergens", "nutrition"]})
            meals = [as_dict(x) for x in store.get("set_meals", {}).values()]
            restaurants[rname] = {"dishes": dishes[:180], "set_meals": meals[:50], "user_orders": str(store.get("user_orders", {}))[:3500]}
        return {"restaurants": restaurants}
    if scenario == "kitchen":
        ingredients = []
        for obj in getattr(db, "ingredients", {}).values():
            r = as_dict(obj)
            ingredients.append({k: r.get(k) for k in ["name", "quantity", "category", "storage_location", "expiry_date", "nutrition"]})
        recipes = []
        for obj in getattr(db, "recipes", {}).values():
            r = as_dict(obj)
            recipes.append({k: r.get(k) for k in ["name", "ingredients", "allergens", "taste", "nutritional_characteristics"]})
        return {"ingredients": ingredients, "recipes": recipes, "menus": str(getattr(db, "user_menus", {}))[:3000], "shopping_lists": str(getattr(db, "user_shopping_lists", {}))[:3000]}
    return {}


def extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


def call_openai(prompt: str, timeout: int = 90) -> Dict[str, Any]:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        return {"error": "missing_OPENAI_API_KEY"}
    base = (os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE") or "https://ai-pixel.online/v1").rstrip("/")
    model = os.environ.get("TRACK2_OPENAI_MODEL") or os.environ.get("OPENAI_MODEL") or "gpt-5.5"
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You produce EgoBench Track2 tool-call programs. Return strict JSON only. Do not use hidden ground truth; use only provided instruction, DB, visual card, and tool signatures."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": int(os.environ.get("TRACK2_OPENAI_MAX_OUTPUT_TOKENS", "3500")),
    }
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        base + "/chat/completions",
        data=data,
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
        method="POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=timeout) as resp:
        obj = json.loads(resp.read().decode("utf-8", errors="replace"))
    content = obj["choices"][0]["message"]["content"]
    parsed = extract_json_object(content)
    parsed["_api_model"] = model
    parsed["_api_base"] = re.sub(r"//.*@", "//", base)
    return parsed


def build_prompt(spec: str, index: int, pos: int, scenario: str, row: Dict[str, Any], db: Any) -> str:
    payload = {
        "task": "Generate 12 diverse candidate tool programs for this EgoBench Track2 task. A program is a list of tool calls. Prefer short GT-like process: required mutation(s) and final aggregate/summary, plus retrieval only when it is needed for branch decisions. Include both plausible visual branches when uncertain.",
        "strict_output_schema": {
            "candidates": [
                {
                    "candidate_id": "short_id",
                    "rationale": "brief non-GT reasoning",
                    "tool_program": [{"tool_name": "exact_tool_name", "parameters": {}}],
                }
            ]
        },
        "hard_rules": [
            "Use exact tool names and parameter names from tool_signatures.",
            "For restaurant/order add_dish_to_order include category, price, tax_rate, discount if available in DB.",
            "For order include restaurant_name on every order tool.",
            "For restaurant/order aggregate dishes[] use dish_name, not product_name.",
            "Mutation target must come from current task instruction/visual card/DB inference, not from ground truth.",
            "Do not ask the user. Do not output natural language outside JSON.",
        ],
        "scenario": scenario,
        "spec": spec,
        "materialized_index": index,
        "instruction": row.get("Instruction", ""),
        "current_value_field": row.get("value"),
        "image_name": row.get("image_name"),
        "qwen_visual_card": qwen_card(spec, pos),
        "tool_signatures": tool_signatures(db),
        "compact_db": compact_db(scenario, db),
    }
    return json.dumps(payload, ensure_ascii=False, default=str)


def synth_one(t: Dict[str, Any]) -> Dict[str, Any]:
    spec = t["spec"]
    scenario = t["scenario"]
    pos = int(t["local_pos"])
    index = int(t["index"])
    number = int(re.sub(r"^\D+", "", spec))
    row = read_json(SPLIT_DIR / f"{spec}.json", [])[pos]
    db = init_db(scenario, number)
    prompt = build_prompt(spec, index, pos, scenario, row, db)
    try:
        obj = call_openai(prompt)
    except Exception as exc:
        obj = {"error": f"{type(exc).__name__}: {exc}"}
    cands = []
    for i, c in enumerate(obj.get("candidates") or []):
        if not isinstance(c, dict):
            continue
        prog = c.get("tool_program") or []
        if not isinstance(prog, list):
            continue
        cands.append({
            "candidate_id": "V24_GPT_" + str(c.get("candidate_id") or i),
            "source": "GPT_PROGRAM_SYNTHESIS",
            "tool_program": prog,
            "risk_flags": [],
            "shape_confidence": 1.1,
            "meta": {"rationale": c.get("rationale", ""), "api_model": obj.get("_api_model")},
        })
    return {
        "spec": spec,
        "index": index,
        "local_pos": pos,
        "scenario": scenario,
        "api_error": obj.get("error", ""),
        "candidate_count": len(cands),
        "candidates": cands,
        "gt_used": False,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default=str(CODEX / "analysis" / "v24_target_tasks.json"))
    ap.add_argument("--output", default=str(CODEX / "analysis" / "v24_gpt_program_candidates.jsonl"))
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    targets = read_json(Path(args.targets), {}).get("targets") or []
    rows: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = [ex.submit(synth_one, t) for t in targets]
        for fut in as_completed(futs):
            rows.append(fut.result())
    rows.sort(key=lambda r: (r["spec"], r["index"]))
    write_jsonl(Path(args.output), rows)
    print(json.dumps({"rows": len(rows), "candidate_count": sum(r.get("candidate_count", 0) for r in rows), "errors": sum(1 for r in rows if r.get("api_error")), "output": args.output}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

