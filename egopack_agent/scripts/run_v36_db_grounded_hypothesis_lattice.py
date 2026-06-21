#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V36 DB-grounded slot hypothesis lattice.

Val41 shadow only. No final run, no final hidden metadata, no V10 overwrite.
Val41 GT is used only by official evaluator after candidate trajectories have
already been generated.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import io
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


CODEX = Path(os.environ.get("CODEX_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/codex"))
EGO = Path(os.environ.get("EGO_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/EgoBench"))
SPLIT_DIR = CODEX / "state" / "materialized_splits" / "validation_A_limit30"
V10_ZIP = CODEX / "submissions" / "V10_full_memory_final_candidate_draft_track2.zip"
V10_EXPECTED_SHA = "5f2b8aacf986e4b3107b560e76fe8dfc52b5b3da6e407c9e0e6d0433db36849d"
V34_RUN_ID = "v34_v32_expansion_20260621_1658"

sys.path.insert(0, str(CODEX / "scripts"))
sys.path.insert(0, str(CODEX / "wrappers"))
sys.path.insert(0, str(CODEX))
sys.path.insert(0, str(EGO))

import run_v32_native_vision_val41 as v32  # noqa: E402
from egobench_agent_plus.v25_evidence_entity_matcher import (  # noqa: E402
    collect_db_entities,
    evidence_mentions,
    match_entities,
)
from egobench_agent_plus.v32_tool_loop_guard import execute_calls  # noqa: E402


RETRIEVAL_PREFIXES = ("get_", "find_", "filter_", "list_")
MUTATION_RE = re.compile(r"^(add|remove|delete|update|modify|clear)_|_(to|from)_(cart|order|shopping_list|menu)$")
AGG_TOOLS = {
    "compute_total_payment",
    "compute_total_tax",
    "compute_total_nutrition",
    "compute_total_nutritions",
    "tally_total_nutritional_characteristics",
    "tally_total_tastes",
    "get_user_order_summary",
    "compute_total",
    "compute_cart_tax",
    "sum_cart_nutrition",
}
V36_VERSION = "V36_DB_GROUNDED_HYPOTHESIS_LATTICE"


def stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return default


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def read_jsonl(path: Path, limit: int = 0) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
            if limit and len(rows) >= limit:
                break
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    tmp.replace(path)


def sha256_file(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def task_key(spec: str, idx: int) -> str:
    return f"{spec}::{int(idx)}"


def norm(v: Any) -> str:
    return re.sub(r"\s+", " ", str(v or "").strip().lower())


def words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9']+", norm(text)) if len(w) > 1}


def sim(a: str, b: str) -> float:
    a = norm(a)
    b = norm(b)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if a in b or b in a:
        return 0.88
    wa, wb = words(a), words(b)
    j = len(wa & wb) / max(1, len(wa | wb))
    return max(j, SequenceMatcher(None, a, b).ratio())


def asdict_safe(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    if isinstance(obj, dict):
        return dict(obj)
    try:
        return dict(getattr(obj, "__dict__", {}) or {})
    except Exception:
        return {}


def load_evidence_cache() -> Dict[str, Dict[str, Any]]:
    cache: Dict[str, Dict[str, Any]] = {}
    for path in [
        CODEX / "analysis" / "v26_mm_evidence_val41.jsonl",
        CODEX / "analysis" / "v25_new_mm_evidence.jsonl",
        CODEX / "analysis" / "v36_mm_evidence_val41.jsonl",
    ]:
        for row in read_jsonl(path):
            key = row.get("task_key")
            if key:
                cache[str(key)] = row
            spec, idx = row.get("spec"), row.get("index")
            if spec is not None and idx is not None:
                cache[f"{spec}::{idx}"] = row
    return cache


def all_tasks_by_key() -> Dict[str, Dict[str, Any]]:
    return {task_key(t["spec"], int(t["index"])): t for t in v32.all_tasks()}


def row_for_task(task: Dict[str, Any]) -> Dict[str, Any]:
    rows = read_json(SPLIT_DIR / f"{task['spec']}.json", [])
    return rows[int(task["local_pos"])]


def make_item_from_program(row: Dict[str, Any], program: List[Dict[str, Any]], results: List[Dict[str, Any]], final_text: str, candidate_id: str) -> Dict[str, Any]:
    return {
        "task_id": row.get("task_id", 1),
        "mode": "text",
        "instruction": row.get("Instruction", ""),
        "image_description": "",
        "dialogue": [{"role": "agent", "turn": 0, "content": final_text}],
        "tool_calls": [{"turn": 0, "calls": program, "results": results}],
        "tool_calls_count": len(program),
        "rounds_count": 1,
        "tokens_consumed": 0,
        "final_run": False,
        "uses_final_hidden_metadata": False,
        "uses_val41_gt_for_policy": False,
        "v36_meta": {"candidate_id": candidate_id, "version": V36_VERSION},
    }


def eval_item(task: Dict[str, Any], row: Dict[str, Any], item: Dict[str, Any]) -> Dict[str, Any]:
    return v32.evaluate_one(row, item, str(task["scenario"]), int(task["number"]))


def aggregate_scores(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    return v32.aggregate(list(rows))


def score_count(summary: Dict[str, Any], total: int = 41) -> int:
    return int(round(float(summary.get("joint", 0) or 0) * total))


def tool_names(program: List[Dict[str, Any]]) -> List[str]:
    return [str(c.get("tool_name") or "") for c in program]


def is_mutation(name: str) -> bool:
    return bool(MUTATION_RE.search(str(name)))


def valid_params(db: Any, tool: str, params: Dict[str, Any]) -> Dict[str, Any]:
    if hasattr(db, tool):
        try:
            sig = inspect.signature(getattr(db, tool))
            return {k: v for k, v in (params or {}).items() if k in sig.parameters}
        except Exception:
            pass
    return params or {}


def call_ok(res: Dict[str, Any]) -> bool:
    if res.get("status") in {"error", "blocked"}:
        return False
    content = str(res.get("content") or "")
    return '"status": "error"' not in content and "not found" not in content.lower()


def db_user_ids(scenario: str, db: Any) -> List[str]:
    if scenario == "retail":
        return sorted(set(getattr(db, "user_carts", {}).keys()) | set(getattr(db, "user_shopping_lists", {}).keys()))
    if scenario == "restaurant":
        return sorted(getattr(db, "user_orders", {}).keys())
    if scenario == "order":
        ids = set()
        for store in getattr(db, "restaurants", {}).values():
            ids.update((store.get("user_orders") or {}).keys())
        return sorted(ids)
    if scenario == "kitchen":
        return sorted(set(getattr(db, "user_menus", {}).keys()) | set(getattr(db, "user_shopping_lists", {}).keys()))
    return []


def init_db_silent(scenario: str, number: int) -> Any:
    old_stdout = sys.stdout
    try:
        sys.stdout = io.StringIO()
        return v32.init_db(scenario, number)
    finally:
        sys.stdout = old_stdout


def extract_user_ids(text: str) -> List[str]:
    ids = []
    for pat in [r"User ID[:\s]+([A-Za-z0-9_]+)", r"user ID is ([A-Za-z0-9_]+)", r"ID[:\s]+([A-Za-z0-9_]+)"]:
        for m in re.finditer(pat, text, flags=re.I):
            ids.append(m.group(1))
    return list(dict.fromkeys(ids))


def get_gt100_skeletons() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for r in read_jsonl(CODEX / "gt_distill_v16" / "gt100_pool.jsonl", limit=200000):
        if not r.get("no_final_metadata", True):
            continue
        if not r.get("excluded_final309", True) or not r.get("excluded_val41", True):
            continue
        calls = r.get("tool_calls") or []
        if not calls:
            continue
        names = tool_names(calls)
        scenario = str(r.get("scenario") or "")
        family = family_for_tools(scenario, names, str(r.get("task_type") or ""), str(r.get("instruction") or ""))
        rows.append({
            "source": "nonfinal_gt100",
            "scenario": scenario,
            "spec": r.get("spec"),
            "task_type": r.get("task_type"),
            "program_family": family,
            "tool_names": names,
            "tool_steps": skeletonize_calls(calls),
            "required_slots": required_slots(calls),
            "has_mutation": any(is_mutation(n) for n in names),
            "has_closure": any(n in AGG_TOOLS for n in names),
            "no_final_metadata": True,
        })
    return rows


def family_for_tools(scenario: str, names: List[str], task_type: str, instruction: str) -> str:
    has_mut = any(is_mutation(n) for n in names)
    has_agg = any(n in AGG_TOOLS for n in names)
    first = names[0] if names else "none"
    if scenario == "retail":
        if "add_to_cart" in names:
            return "retail_product_branch_add_cart" if any(n.startswith(("get_", "find_")) for n in names[:3]) else "retail_direct_add_cart"
        if "compute_total_payment" in names or "compute_total_tax" in names or "compute_total_nutrition" in names:
            return "retail_query_then_payment"
        return "retail_query_only"
    if scenario == "order":
        if any("set_meal" in n for n in names):
            return "order_setmeal_details_add_payment"
        if "add_dish_to_order" in names:
            return "order_restaurant_pin_dish_add_payment"
        return "order_query_or_summary"
    if scenario == "restaurant":
        if any("set_meal" in n for n in names):
            return "restaurant_setmeal_query_mutate_closure"
        if "add_dish_to_order" in names:
            return "restaurant_menu_dish_query_mutate"
        return "restaurant_menu_dish_query_answer"
    if scenario == "kitchen":
        if "add_recipe_to_menu" in names and has_agg:
            return "kitchen_recipe_branch_add_menu_aggregate"
        if "add_to_shopping_list" in names and has_agg:
            return "kitchen_shopping_list_aggregate"
        if has_mut:
            return "kitchen_mutation_closure"
        return f"kitchen_query_{first}"
    return f"{scenario}_{'mutation' if has_mut else 'query'}_{'aggregate' if has_agg else 'simple'}"


def skeletonize_calls(calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    slot_keys = {
        "user_id", "product_name", "dish_name", "set_meal_name", "restaurant_name",
        "recipe_name", "ingredient_name", "category", "taste", "tag", "characteristic",
        "allergen", "min_price", "max_price", "quantity", "qty", "recipes", "products", "dishes", "ingredients",
    }
    for call in calls:
        params = call.get("parameters") or {}
        sk_params = {}
        for k, v in params.items():
            sk_params[k] = f"<{k}>" if k in slot_keys else "<value>"
        out.append({"tool_name": call.get("tool_name"), "parameters": sk_params})
    return out


def required_slots(calls: List[Dict[str, Any]]) -> List[str]:
    slots = set()
    for call in calls:
        for k in (call.get("parameters") or {}).keys():
            slots.add(str(k))
    return sorted(slots)


def extract_instruction_tokens(text: str) -> Dict[str, Any]:
    t = norm(text)
    attrs = []
    for a in [
        "low_sugar", "low sugar", "high protein", "high_protein", "low fat", "low_fat",
        "high sodium", "high_sodium", "gluten_free", "gluten-free", "vegan", "spicy",
        "salty", "sweet", "sour", "bitter", "umami", "fresh", "rich", "buttery",
    ]:
        if a in t:
            attrs.append(a.replace(" ", "_").replace("-", "_"))
    categories = []
    for c in [
        "wine", "sandwich", "sandwiches", "panini", "cold cut", "cold cuts", "snack",
        "pizza", "pasta", "salads", "fruit", "fruits", "vegetable", "vegetables",
        "seasoning", "staple", "dry goods", "meat", "fish",
    ]:
        if c in t:
            categories.append(c)
    rank = ""
    if any(x in t for x in ["lowest", "cheapest", "fewest"]):
        rank = "lowest"
    if any(x in t for x in ["highest", "most expensive", "highest total", "highest protein"]):
        rank = "highest"
    return {"attributes": sorted(set(attrs)), "categories": sorted(set(categories)), "rank": rank}


def entity_score(name: str, meta: Dict[str, Any], text: str, mentions: List[str], instr: Dict[str, Any]) -> float:
    s = 0.0
    n = norm(name)
    if n in norm(text):
        s += 8.0
    for m in mentions:
        ms = sim(n, m)
        if ms >= 0.82:
            s += 5.0 * ms
        elif ms >= 0.45:
            s += 2.0 * ms
    meta_text = json.dumps(meta, ensure_ascii=False).lower()
    for cat in instr["categories"]:
        c = cat.replace("sandwich", "sandwiches").replace("cold cut", "cold cuts")
        if c in meta_text:
            s += 3.0
    for attr in instr["attributes"]:
        if attr in meta_text:
            s += 2.0
    return round(s, 4)


def build_lattice(task: Dict[str, Any], row: Dict[str, Any], evidence: Dict[str, Any], db: Any) -> Dict[str, Any]:
    scenario = str(task["scenario"])
    instruction = str(row.get("Instruction") or "")
    mentions = evidence_mentions(evidence)
    mentions.append(instruction)
    instr = extract_instruction_tokens(instruction + "\n" + json.dumps(evidence.get("candidate_slots", {}), ensure_ascii=False))
    matches = match_entities(scenario, db, evidence, top_k=12)
    entities = collect_db_entities(scenario, db)
    user_ids = extract_user_ids(instruction)
    for uid in db_user_ids(scenario, db):
        if uid not in user_ids:
            user_ids.append(uid)
    slots: Dict[str, List[Dict[str, Any]]] = {"user_id": [{"value": u, "score": 10 if u in instruction else 2, "source": "instruction_or_db"} for u in user_ids[:8]]}
    for typ, rows in entities.items():
        scored = []
        seen = set()
        for m in matches.get(typ, []):
            name = str(m.get("canonical_name") or "")
            if name and name not in seen:
                seen.add(name)
                scored.append({"value": name, "score": float(m.get("score", 0)) * 10 + 1, "source": "evidence_match", "meta": m.get("meta") or {}})
        for e in rows:
            name = str(e.get("canonical_name") or "")
            if not name or name in seen:
                continue
            meta = e.get("meta") or {}
            sc = entity_score(name, meta, instruction, mentions, instr)
            if sc > 0.3:
                seen.add(name)
                scored.append({"value": name, "score": sc, "source": "db_fuzzy", "meta": meta})
        scored.sort(key=lambda x: x["score"], reverse=True)
        limit = {"product": 30, "dish": 30, "set_meal": 20, "ingredient": 30, "recipe": 20, "restaurant": 10, "category": 12}.get(typ, 10)
        slots[typ] = scored[:limit]
    lattice = {
        "task_key": task_key(task["spec"], int(task["index"])),
        "scenario": scenario,
        "instruction": instruction,
        "intent": infer_intent(instruction, scenario),
        "instruction_features": instr,
        "slots": slots,
        "evidence_sources": list(evidence.keys()),
        "no_final_metadata": True,
    }
    return lattice


def infer_intent(instruction: str, scenario: str) -> Dict[str, Any]:
    t = norm(instruction)
    mutation = []
    if "add" in t:
        mutation.append("add")
    if "remove" in t or "delete" in t:
        mutation.append("remove")
    if "update" in t or "change" in t or "replace" in t:
        mutation.append("update")
    closure = []
    if "payment" in t or "payable" in t or "total amount" in t or "total cost" in t:
        closure.append("payment")
    if "tax" in t:
        closure.append("tax")
    if "nutrition" in t or "nutritional" in t or "calorie" in t or "protein" in t or "fat content" in t or "sugar content" in t:
        closure.append("nutrition")
    if "summary" in t or "summarize" in t:
        closure.append("summary")
    return {
        "mutation": mutation,
        "closure": sorted(set(closure)),
        "branch": " if " in f" {t} " or "determine" in t or "judge" in t or "check if" in t,
        "query_only": not mutation,
    }


def tool_arg_candidates(lattice: Dict[str, Any], scenario: str) -> Dict[str, List[Any]]:
    slots = lattice["slots"]
    arg: Dict[str, List[Any]] = {}
    arg["user_id"] = [x["value"] for x in slots.get("user_id", [])[:4]]
    arg["product_name"] = [x["value"] for x in slots.get("product", [])[:10]]
    arg["dish_name"] = [x["value"] for x in slots.get("dish", [])[:12]]
    arg["set_meal_name"] = [x["value"] for x in slots.get("set_meal", [])[:8]]
    arg["restaurant_name"] = [x["value"] for x in slots.get("restaurant", [])[:5]]
    arg["recipe_name"] = [x["value"] for x in slots.get("recipe", [])[:10]]
    arg["ingredient_name"] = [x["value"] for x in slots.get("ingredient", [])[:12]]
    arg["category"] = [x["value"] for x in slots.get("category", [])[:8]]
    attrs = lattice["instruction_features"]["attributes"]
    arg["taste"] = [a.replace("_", " ") for a in attrs if a in {"spicy", "salty", "sweet", "sour", "bitter", "umami", "fresh", "rich", "buttery"}][:4]
    arg["tag"] = attrs[:8]
    arg["characteristic"] = attrs[:8]
    arg["allergen"] = [x for x in ["gluten", "soy", "fish", "dairy", "nuts"] if x in norm(lattice["instruction"])]
    arg["qty"] = [1]
    arg["quantity"] = [1, 100, 200, 300, 500]
    arg["min_price"] = [0]
    arg["max_price"] = [20, 50, 60, 100000]
    return arg


def fill_template_call(template: Dict[str, Any], args: Dict[str, List[Any]], db: Any) -> List[Dict[str, Any]]:
    tool = str(template.get("tool_name") or "")
    params = template.get("parameters") or {}
    variants: List[Dict[str, Any]] = []
    if not hasattr(db, tool):
        return variants
    sig = inspect.signature(getattr(db, tool))
    base: Dict[str, Any] = {}
    flexible_keys = []
    for k in sig.parameters:
        vals = args.get(k) or []
        if not vals:
            if k in {"products", "dishes", "ingredients"}:
                vals = [[]]
            elif k in {"recipes"}:
                vals = [args.get("recipe_name", [])[:4]]
            elif k == "included_dishes":
                vals = [[]]
            elif k in {"new_price", "new_discount"}:
                vals = [1]
            else:
                return []
        base[k] = vals[0]
        if k in {"product_name", "dish_name", "set_meal_name", "recipe_name", "ingredient_name", "category", "restaurant_name"}:
            flexible_keys.append(k)
    variants.append({"tool_name": tool, "parameters": dict(base)})
    for k in flexible_keys[:2]:
        for v in (args.get(k) or [])[1:4]:
            p = dict(base)
            p[k] = v
            variants.append({"tool_name": tool, "parameters": p})
    return variants[:6]


def generic_query_calls(scenario: str, lattice: Dict[str, Any], db: Any) -> List[List[Dict[str, Any]]]:
    args = tool_arg_candidates(lattice, scenario)
    calls: List[List[Dict[str, Any]]] = []
    user = (args.get("user_id") or [""])[0]
    instr = lattice["instruction_features"]
    intent = lattice["intent"]
    if scenario == "retail":
        products = args.get("product_name", [])[:8]
        for p in products:
            chain = []
            for tool in ["get_nutrition", "get_price", "get_discount", "get_tax_rate", "get_category"]:
                if hasattr(db, tool):
                    chain.append({"tool_name": tool, "parameters": {"product_name": p}})
            if "add" in intent["mutation"] and user:
                chain.append({"tool_name": "add_to_cart", "parameters": {"user_id": user, "product_name": p, "qty": 1}})
            chain.extend(retail_closure(user, products[:3], intent))
            calls.append(chain)
    elif scenario in {"restaurant", "order"}:
        restaurants = args.get("restaurant_name", []) or [""]
        dishes = args.get("dish_name", [])[:10]
        set_meals = args.get("set_meal_name", [])[:6]
        prefix = lambda params: {"restaurant_name": restaurants[0], **params} if scenario == "order" and restaurants[0] else params
        for d in dishes:
            chain = []
            for tool in ["get_dish_price", "get_dish_nutrition", "get_dish_taste_profile", "get_dish_discount", "get_tax_rate", "get_dish_allergens"]:
                if hasattr(db, tool):
                    chain.append({"tool_name": tool, "parameters": prefix({"dish_name": d})})
            if "add" in intent["mutation"] and user:
                chain.append({"tool_name": "add_dish_to_order", "parameters": prefix({"user_id": user, "dish_name": d, "quantity": 1})})
            chain.extend(food_closure(scenario, restaurants[0], user, dishes[:3], set_meals[:2], intent))
            calls.append(chain)
        for sm in set_meals:
            chain = []
            if hasattr(db, "get_set_meal_details"):
                chain.append({"tool_name": "get_set_meal_details", "parameters": prefix({"set_meal_name": sm})})
            if "add" in intent["mutation"] and user and hasattr(db, "add_set_meal_to_order"):
                chain.append({"tool_name": "add_set_meal_to_order", "parameters": prefix({"user_id": user, "set_meal_name": sm, "quantity": 1})})
            chain.extend(food_closure(scenario, restaurants[0], user, dishes[:2], [sm], intent))
            calls.append(chain)
    elif scenario == "kitchen":
        recipes = args.get("recipe_name", [])[:10]
        ingredients = args.get("ingredient_name", [])[:10]
        for r in recipes:
            chain = []
            for tool in ["get_recipe_ingredients", "get_recipe_nutritional_characteristics", "get_recipe_allergens", "get_recipe_taste", "get_cooking_steps"]:
                if hasattr(db, tool):
                    chain.append({"tool_name": tool, "parameters": {"recipe_name": r}})
            if "add" in intent["mutation"] and user:
                chain.append({"tool_name": "add_recipe_to_menu", "parameters": {"user_id": user, "recipe_name": r}})
            chain.extend(kitchen_closure(user, recipes[:4], ingredients[:4], intent))
            calls.append(chain)
        for ing in ingredients:
            chain = []
            for tool in ["get_ingredient_nutrition", "get_ingredient_quantity", "get_ingredient_shelf_life", "find_ingredient_category"]:
                if hasattr(db, tool):
                    chain.append({"tool_name": tool, "parameters": {"ingredient_name": ing}})
            if "add" in intent["mutation"] and user:
                chain.append({"tool_name": "add_to_shopping_list", "parameters": {"user_id": user, "ingredient_name": ing, "quantity": 1}})
            chain.extend(kitchen_closure(user, recipes[:3], [ing], intent))
            calls.append(chain)
    return [dedup_program(c) for c in calls if c]


def retail_closure(user: str, products: List[str], intent: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    prod_objs = [{"product_name": p, "quantity": 1} for p in products]
    if user and ("payment" in intent["closure"] or "summary" in intent["closure"]):
        out.append({"tool_name": "compute_total_payment", "parameters": {"user_id": user, "products": prod_objs}})
    if user and "tax" in intent["closure"]:
        out.append({"tool_name": "compute_total_tax", "parameters": {"user_id": user, "products": prod_objs}})
    if user and "nutrition" in intent["closure"]:
        out.append({"tool_name": "compute_total_nutrition", "parameters": {"user_id": user, "products": prod_objs}})
    return out


def food_closure(scenario: str, restaurant: str, user: str, dishes: List[str], set_meals: List[str], intent: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not user:
        return []
    base = {"user_id": user}
    if scenario == "order" and restaurant:
        base["restaurant_name"] = restaurant
    dish_objs = [{"dish_name": d, "quantity": 1} for d in dishes]
    out = []
    if "summary" in intent["closure"] or "payment" in intent["closure"] or intent["mutation"]:
        out.append({"tool_name": "get_user_order_summary", "parameters": dict(base)})
    if "payment" in intent["closure"]:
        out.append({"tool_name": "compute_total_payment", "parameters": {**base, "dishes": dish_objs}})
    if "tax" in intent["closure"]:
        out.append({"tool_name": "compute_total_tax", "parameters": {**base, "dishes": dish_objs}})
    if "nutrition" in intent["closure"]:
        out.append({"tool_name": "compute_total_nutrition", "parameters": {**base, "dishes": dish_objs}})
    return out


def kitchen_closure(user: str, recipes: List[str], ingredients: List[str], intent: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not user:
        return []
    out = []
    if "summary" in intent["closure"] or "nutrition" in intent["closure"]:
        if recipes:
            out.append({"tool_name": "tally_total_nutritional_characteristics", "parameters": {"user_id": user, "recipes": recipes}})
        if ingredients:
            out.append({"tool_name": "compute_total_nutritions", "parameters": {"user_id": user, "ingredients": [{"ingredient_name": i, "quantity": 1} for i in ingredients]}})
    return out


def dedup_program(program: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    seen = set()
    for c in program:
        key = json.dumps(c, sort_keys=True, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def instantiate_from_skeletons(task: Dict[str, Any], lattice: Dict[str, Any], db: Any, skeletons: List[Dict[str, Any]], max_candidates: int) -> List[Dict[str, Any]]:
    scenario = str(task["scenario"])
    args = tool_arg_candidates(lattice, scenario)
    cands: List[Dict[str, Any]] = []
    relevant = [s for s in skeletons if s.get("scenario") == scenario]
    relevant.sort(key=lambda s: skeleton_relevance(s, lattice), reverse=True)
    for sk in relevant[:12]:
        chains: List[List[Dict[str, Any]]] = [[]]
        for tmpl in sk.get("tool_steps", [])[:10]:
            variants = fill_template_call(tmpl, args, db)
            if not variants:
                continue
            new_chains = []
            for ch in chains:
                for v in variants[:3]:
                    new_chains.append(ch + [v])
            chains = new_chains[:24]
        for chain in chains[:20]:
            if not chain:
                continue
            chain = dedup_program(chain)
            if unsafe_program(chain):
                continue
            cands.append({"source": "skeleton", "program_family": sk.get("program_family"), "tool_program": chain})
            if len(cands) >= max_candidates:
                return cands
    for chain in generic_query_calls(scenario, lattice, db):
        if not unsafe_program(chain):
            cands.append({"source": "lattice_generic", "program_family": f"{scenario}_generic_lattice", "tool_program": chain})
        if len(cands) >= max_candidates:
            break
    return cands


def skeleton_relevance(sk: Dict[str, Any], lattice: Dict[str, Any]) -> float:
    s = 0.0
    intent = lattice["intent"]
    if intent["mutation"] and sk.get("has_mutation"):
        s += 5
    if intent["closure"] and sk.get("has_closure"):
        s += 4
    fam = str(sk.get("program_family") or "")
    instr = norm(lattice["instruction"])
    for tok in words(fam.replace("_", " ")):
        if tok in instr:
            s += 0.5
    return s


def unsafe_program(program: List[Dict[str, Any]]) -> bool:
    names = tool_names(program)
    if not names:
        return True
    if names[0].startswith(("find_products_by_price_range", "get_all_")):
        return True
    if any(is_mutation(n) for n in names[:1]):
        return True
    return False


def augment_existing_program(task: Dict[str, Any], row: Dict[str, Any], base_item: Dict[str, Any], lattice: Dict[str, Any], db: Any, max_aug: int = 30) -> List[Dict[str, Any]]:
    base_program = []
    for entry in base_item.get("tool_calls") or []:
        base_program.extend(entry.get("calls") or [])
    if not base_program:
        return []
    scenario = str(task["scenario"])
    args = tool_arg_candidates(lattice, scenario)
    augment_calls: List[Dict[str, Any]] = []
    names = set(tool_names(base_program))
    if scenario == "retail":
        for p in args.get("product_name", [])[:8]:
            for tool in ["get_nutrition", "get_price", "get_discount", "get_tax_rate", "get_category"]:
                if hasattr(db, tool):
                    augment_calls.append({"tool_name": tool, "parameters": {"product_name": p}})
    elif scenario in {"restaurant", "order"}:
        r = (args.get("restaurant_name") or [""])[0]
        prefix = lambda params: {"restaurant_name": r, **params} if scenario == "order" and r else params
        for d in args.get("dish_name", [])[:8]:
            for tool in ["get_dish_price", "get_dish_nutrition", "get_dish_taste_profile", "get_dish_discount", "get_tax_rate", "get_dish_allergens"]:
                if hasattr(db, tool):
                    augment_calls.append({"tool_name": tool, "parameters": prefix({"dish_name": d})})
        for sm in args.get("set_meal_name", [])[:4]:
            if hasattr(db, "get_set_meal_details"):
                augment_calls.append({"tool_name": "get_set_meal_details", "parameters": prefix({"set_meal_name": sm})})
    elif scenario == "kitchen":
        for rcp in args.get("recipe_name", [])[:6]:
            for tool in ["get_recipe_ingredients", "get_recipe_nutritional_characteristics", "get_recipe_allergens", "get_recipe_taste", "get_cooking_steps"]:
                if hasattr(db, tool):
                    augment_calls.append({"tool_name": tool, "parameters": {"recipe_name": rcp}})
        for ing in args.get("ingredient_name", [])[:6]:
            for tool in ["get_ingredient_nutrition", "get_ingredient_quantity", "get_ingredient_shelf_life", "find_ingredient_category"]:
                if hasattr(db, tool):
                    augment_calls.append({"tool_name": tool, "parameters": {"ingredient_name": ing}})
    out = []
    for i in range(0, min(len(augment_calls), max_aug), 4):
        prefix = augment_calls[i:i + 4]
        chain = dedup_program(prefix + base_program)
        if not unsafe_program(chain):
            out.append({"source": "v34_augmented_prefix", "program_family": f"{scenario}_v34_nonmutating_prefix", "tool_program": chain})
        chain2 = dedup_program(base_program + prefix)
        if not unsafe_program(chain2):
            out.append({"source": "v34_augmented_suffix", "program_family": f"{scenario}_v34_nonmutating_suffix", "tool_program": chain2})
    return out[:20]


def dry_run_candidate(task: Dict[str, Any], row: Dict[str, Any], cand: Dict[str, Any], lattice: Dict[str, Any], idx: int) -> Dict[str, Any]:
    db = init_db_silent(str(task["scenario"]), int(task["number"]))
    program = []
    for call in cand["tool_program"]:
        tool = str(call.get("tool_name") or "")
        params = valid_params(db, tool, call.get("parameters") or {})
        program.append({"tool_name": tool, "parameters": params})
    program = dedup_program(program)
    results = execute_calls(db, program)
    verifier = verify_program(task, lattice, program, results)
    cid = f"V36::{cand.get('source')}::{cand.get('program_family')}::{task_key(task['spec'], task['index'])}::{idx}"
    final = final_text_from_results(program, results, verifier)
    item = make_item_from_program(row, program, results, final, cid)
    score = eval_item(task, row, item)
    return {
        "candidate_id": cid,
        "task_key": task_key(task["spec"], int(task["index"])),
        "spec": task["spec"],
        "index": int(task["index"]),
        "local_pos": int(task["local_pos"]),
        "scenario": task["scenario"],
        "number": int(task["number"]),
        "source": cand.get("source"),
        "program_family": cand.get("program_family"),
        "tool_program": program,
        "results": results,
        "verifier": verifier,
        "score": score,
        "item": item,
        "no_final_metadata": True,
    }


def verify_program(task: Dict[str, Any], lattice: Dict[str, Any], program: List[Dict[str, Any]], results: List[Dict[str, Any]]) -> Dict[str, Any]:
    names = tool_names(program)
    intent = lattice["intent"]
    score = 0.0
    flags = []
    if names:
        score += min(20, len(names) * 1.5)
    if any(is_mutation(n) for n in names):
        score += 14
    elif intent["mutation"]:
        flags.append("mutation_missing")
        score -= 8
    if any(n in AGG_TOOLS for n in names):
        score += 10
    elif intent["closure"]:
        flags.append("closure_missing")
        score -= 8
    if names and any(n.startswith(RETRIEVAL_PREFIXES) or n in AGG_TOOLS for n in names[:3]):
        score += 8
    else:
        flags.append("retrieval_prefix_missing")
        score -= 12
    err = sum(1 for r in results if not call_ok(r))
    if err:
        flags.append(f"tool_errors={err}")
        score -= err * 6
    if unsafe_program(program):
        flags.append("unsafe_program")
        score -= 20
    if repeated_mutation(program):
        flags.append("duplicate_mutation")
        score -= 12
    nonempty = sum(1 for r in results if result_nonempty(r))
    score += min(15, nonempty * 1.2)
    return {"score": round(score, 4), "risk_flags": flags, "nonempty_results": nonempty, "tool_errors": err}


def result_nonempty(r: Dict[str, Any]) -> bool:
    if not call_ok(r):
        return False
    content = str(r.get("content") or "")
    return content not in {"{}", "[]", "null", "\"\""} and "[]" not in content[:20]


def repeated_mutation(program: List[Dict[str, Any]]) -> bool:
    seen = set()
    for c in program:
        if not is_mutation(str(c.get("tool_name") or "")):
            continue
        key = json.dumps(c, sort_keys=True, ensure_ascii=False)
        if key in seen:
            return True
        seen.add(key)
    return False


def final_text_from_results(program: List[Dict[str, Any]], results: List[Dict[str, Any]], verifier: Dict[str, Any]) -> str:
    tail = []
    for c, r in list(zip(program, results))[-4:]:
        if call_ok(r):
            tail.append(f"{c.get('tool_name')} completed")
    if not tail:
        tail = ["Completed requested tool checks."]
    return "; ".join(tail) + f". Verifier score {verifier.get('score')}."


def load_selected_v34_items() -> Tuple[Path, Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    state = read_json(CODEX / "state" / "latest_v34_v32_expansion.json", {})
    result_dir = Path(state.get("selected_conservative", {}).get("result_dir") or "")
    eval_rows, _ = v32.eval_result_dir(result_dir)
    eval_map = {task_key(r["spec"], int(r["index"])): r for r in eval_rows}
    item_map: Dict[str, Dict[str, Any]] = {}
    tasks = all_tasks_by_key()
    for key, task in tasks.items():
        item = v32.load_item(result_dir, str(task["spec"]), int(task["local_pos"]))
        if item:
            item_map[key] = item
    return result_dir, eval_map, item_map


def build_result_dir(result_dir: Path, item_by_key: Dict[Tuple[str, int], Dict[str, Any]], fallback_dir: Path) -> Dict[str, Any]:
    v32.write_result_dir(result_dir, item_by_key, fallback_dir)
    rows, summary = v32.eval_result_dir(result_dir)
    return {"rows": rows, "summary": summary}


def choose_frozen(cands: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    safe = [c for c in cands if "unsafe_program" not in c["verifier"]["risk_flags"] and c["verifier"]["tool_errors"] == 0]
    if not safe:
        safe = cands
    if not safe:
        return None
    return max(safe, key=lambda c: (c["verifier"]["score"], c["score"].get("result", 0), c["score"].get("matches", 0), -c["score"].get("interaction_calls", 9999)))


def choose_oracle(cands: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    if not cands:
        return None
    return max(cands, key=lambda c: (c["score"].get("joint", 0), c["score"].get("tool", 0), c["score"].get("result", 0), c["score"].get("matches", 0), c["verifier"]["score"], -c["score"].get("interaction_calls", 9999)))


def report_table_row(name: str, s: Dict[str, Any]) -> str:
    return f"| {name} | {s.get('valid', 0)} | {score_count(s)}/41 ({s.get('joint',0)*100:.2f}%) | {s.get('result',0)*100:.2f}% | {s.get('tool',0)*100:.2f}% | {s.get('matched_tools',0)}/{s.get('gt_tools',0)} | {s.get('micro',0):.4f} | {s.get('interaction_calls',0)} |"


def write_reports(run_id: str, state: Dict[str, Any], lattice_rows: List[Dict[str, Any]], skeletons: List[Dict[str, Any]], cand_rows: List[Dict[str, Any]], selected_rows: List[Dict[str, Any]], oracle_rows: List[Dict[str, Any]]) -> None:
    reports = CODEX / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    table = [
        "| metric set | valid | joint | result | tool | matched/gt | micro | calls |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        report_table_row("V34_selected", state["baseline"]["V34_selected"]),
        report_table_row("V36_frozen_selected", state["frozen_selected"]["summary"]),
        report_table_row("V36_oracle_best", state["oracle_best"]["summary"]),
    ]
    (reports / f"V36_DATA_USAGE_AND_COMPLIANCE_{run_id}.md").write_text("\n".join([
        f"# V36 Data Usage And Compliance {run_id}",
        "",
        f"- final run: false",
        f"- final hidden metadata used: false",
        f"- val41 GT used for runtime: false",
        f"- val41 GT used for post-eval/oracle diagnostic: true",
        f"- V10 zip exists: {state['preflight']['v10_zip_exists']}",
        f"- V10 sha256: `{state['preflight']['v10_sha256']}`",
        f"- V10 sha expected match: {state['preflight']['v10_sha256'] == V10_EXPECTED_SHA}",
        f"- V10 zip overwritten: {state['preflight']['v10_zip_overwritten']}",
        f"- protected V34 success count: {state['protected_success_count']}",
        f"- target V34 failed count: {state['target_task_count']}",
        f"- non-final skeletons: {len(skeletons)}",
    ]) + "\n", encoding="utf-8")
    scenario_lattice = Counter(r["scenario"] for r in lattice_rows)
    slot_counts = Counter()
    for r in lattice_rows:
        for k, v in r.get("slots", {}).items():
            slot_counts[k] += len(v)
    (reports / f"V36_SLOT_LATTICE_AUDIT_{run_id}.md").write_text("\n".join([
        f"# V36 Slot Lattice Audit {run_id}",
        "",
        f"- lattice tasks: {len(lattice_rows)}",
        f"- by scenario: `{dict(scenario_lattice)}`",
        f"- slot counts: `{dict(slot_counts)}`",
        f"- output: `analysis/v36_slot_hypothesis_lattice.jsonl`",
    ]) + "\n", encoding="utf-8")
    fam_counts = Counter(s["program_family"] for s in skeletons)
    (reports / f"V36_SKELETON_LIBRARY_{run_id}.md").write_text("\n".join([
        f"# V36 Skeleton Library {run_id}",
        "",
        f"- skeleton count: {len(skeletons)}",
        f"- non-final GT100 only: true",
        f"- final metadata used: false",
        "",
        "## Top Families",
        *[f"- {k}: {v}" for k, v in fam_counts.most_common(30)],
        "",
        "- output: `analysis/v36_tool_skeleton_library.jsonl`",
    ]) + "\n", encoding="utf-8")
    src_counts = Counter(c["source"] for c in cand_rows)
    joint_by_src = Counter(c["source"] for c in cand_rows if c["score"].get("joint"))
    (reports / f"V36_EXECUTION_RESULT_{run_id}.md").write_text("\n".join([
        f"# V36 Execution Result {run_id}",
        "",
        f"- candidates executed: {len(cand_rows)}",
        f"- candidates by source: `{dict(src_counts)}`",
        f"- joint candidates by source: `{dict(joint_by_src)}`",
        f"- execution trace: `analysis/v36_execution_trace.jsonl`",
    ]) + "\n", encoding="utf-8")
    new_frozen = state["frozen_selected"].get("new_joint_vs_v34") or []
    (reports / f"V36_PROTECTED_MERGE_RESULT_{run_id}.md").write_text("\n".join([
        f"# V36 Protected Merge Result {run_id}",
        "",
        *table,
        "",
        f"- V36 exceeds V34 11/41: {score_count(state['frozen_selected']['summary']) > 11}",
        f"- new frozen joints vs V34: {new_frozen or 'none'}",
        "- V34 successful tasks protected: true",
    ]) + "\n", encoding="utf-8")
    new_oracle = state["oracle_best"].get("new_joint_vs_v34") or []
    (reports / f"V36_ORACLE_BEST_DIAGNOSTIC_{run_id}.md").write_text("\n".join([
        f"# V36 Oracle Best Diagnostic {run_id}",
        "",
        *table,
        "",
        f"- oracle-best exceeds 11/41: {score_count(state['oracle_best']['summary']) > 11}",
        f"- oracle new joints vs V34: {new_oracle or 'none'}",
        "- oracle uses val41 GT only after candidates are generated: true",
    ]) + "\n", encoding="utf-8")
    fail_counts = Counter()
    useful = Counter()
    for c in oracle_rows:
        if c["score"].get("joint"):
            useful[c.get("program_family") or c.get("source")] += 1
    for c in cand_rows:
        if not c["score"].get("joint"):
            flags = c["verifier"].get("risk_flags") or ["entity_or_branch_wrong"]
            fail_counts[flags[0]] += 1
    (reports / f"V36_NEXT_DECISION_{run_id}.md").write_text("\n".join([
        f"# V36 Next Decision {run_id}",
        "",
        *table,
        "",
        "## Required Answers",
        "",
        "- final run: false",
        "- final hidden metadata used: false",
        f"- V10 sha256 preserved: {state['preflight']['v10_sha256'] == V10_EXPECTED_SHA and not state['preflight']['v10_zip_overwritten']}",
        f"- V36 frozen exceeds V34 11/41: {score_count(state['frozen_selected']['summary']) > 11}",
        f"- V36 oracle-best exceeds 11/41: {score_count(state['oracle_best']['summary']) > 11}",
        f"- new frozen joints: {new_frozen or 'none'}",
        f"- useful skeleton families: `{dict(useful)}`",
        f"- leading failure categories: `{dict(fail_counts.most_common(12))}`",
        f"- decision: {state['decision']}",
    ]) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="v36_db_lattice_" + stamp())
    ap.add_argument("--max-candidates", type=int, default=220)
    ap.add_argument("--task-limit", type=int, default=0)
    args = ap.parse_args()

    run_id = args.run_id
    run_dir = CODEX / "runs" / V36_VERSION / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    before_mtime = V10_ZIP.stat().st_mtime_ns if V10_ZIP.exists() else None
    v10_sha = sha256_file(V10_ZIP)
    evidence_cache = load_evidence_cache()
    tasks = all_tasks_by_key()
    v34_dir, v34_eval, v34_items = load_selected_v34_items()
    v34_summary = aggregate_scores(v34_eval.values())
    protected_keys = {k for k, r in v34_eval.items() if float(r.get("joint", 0)) >= 1.0}
    target_keys = [k for k in tasks if k not in protected_keys]
    if args.task_limit:
        target_keys = target_keys[: args.task_limit]
    skeletons = get_gt100_skeletons()

    lattice_rows: List[Dict[str, Any]] = []
    candidate_rows: List[Dict[str, Any]] = []
    by_task: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    start = time.time()
    for ti, key in enumerate(target_keys, 1):
        task = tasks[key]
        row = row_for_task(task)
        db = init_db_silent(str(task["scenario"]), int(task["number"]))
        evidence = evidence_cache.get(key) or evidence_cache.get(f"{task['spec']}::{task['local_pos']}") or {"utterance": row.get("Instruction", "")}
        lattice = build_lattice(task, row, evidence, db)
        lattice_rows.append(lattice)
        cands = []
        base_item = v34_items.get(key)
        if base_item:
            cands.extend(augment_existing_program(task, row, base_item, lattice, db))
        cands.extend(instantiate_from_skeletons(task, lattice, db, skeletons, max(args.max_candidates - len(cands), 20)))
        seen = set()
        unique = []
        for c in cands:
            sig = json.dumps(c.get("tool_program"), sort_keys=True, ensure_ascii=False)
            if sig not in seen:
                seen.add(sig)
                unique.append(c)
        for ci, cand in enumerate(unique[: args.max_candidates]):
            rec = dry_run_candidate(task, row, cand, lattice, ci)
            candidate_rows.append(rec)
            by_task[key].append(rec)
        if ti % 5 == 0 or ti == len(target_keys):
            elapsed = time.time() - start
            print(f"[{time.strftime('%H:%M:%S')}] V36 {ti}/{len(target_keys)} tasks, candidates={len(candidate_rows)}, elapsed={elapsed:.1f}s", flush=True)

    frozen_choice: Dict[str, Dict[str, Any]] = {}
    oracle_choice: Dict[str, Dict[str, Any]] = {}
    selection_rows: List[Dict[str, Any]] = []
    oracle_rows: List[Dict[str, Any]] = []
    for key in target_keys:
        f = choose_frozen(by_task.get(key, []))
        o = choose_oracle(by_task.get(key, []))
        if f:
            frozen_choice[key] = f
            selection_rows.append({k: v for k, v in f.items() if k not in {"item", "results"}} | {"selected_mode": "frozen"})
        if o:
            oracle_choice[key] = o
            oracle_rows.append({k: v for k, v in o.items() if k not in {"item", "results"}} | {"selected_mode": "oracle"})

    frozen_items: Dict[Tuple[str, int], Dict[str, Any]] = {}
    oracle_items: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for key, task in tasks.items():
        spec, idx = str(task["spec"]), int(task["index"])
        if key in protected_keys:
            item = v34_items.get(key)
            if item:
                frozen_items[(spec, idx)] = item
                oracle_items[(spec, idx)] = item
        else:
            if key in frozen_choice:
                frozen_items[(spec, idx)] = frozen_choice[key]["item"]
            elif key in v34_items:
                frozen_items[(spec, idx)] = v34_items[key]
            if key in oracle_choice:
                oracle_items[(spec, idx)] = oracle_choice[key]["item"]
            elif key in v34_items:
                oracle_items[(spec, idx)] = v34_items[key]

    frozen_dir = EGO / "results" / f"V36_frozen_selected-{run_id}"
    oracle_dir = EGO / "results" / f"V36_oracle_best-{run_id}"
    frozen_eval = build_result_dir(frozen_dir, frozen_items, v34_dir)
    oracle_eval = build_result_dir(oracle_dir, oracle_items, v34_dir)
    v34_success = {k for k, r in v34_eval.items() if r.get("joint")}
    frozen_eval_map = {task_key(r["spec"], int(r["index"])): r for r in frozen_eval["rows"]}
    oracle_eval_map = {task_key(r["spec"], int(r["index"])): r for r in oracle_eval["rows"]}
    frozen_new = sorted(k for k, r in frozen_eval_map.items() if r.get("joint") and k not in v34_success)
    oracle_new = sorted(k for k, r in oracle_eval_map.items() if r.get("joint") and k not in v34_success)

    write_jsonl(CODEX / "analysis" / "v36_slot_hypothesis_lattice.jsonl", lattice_rows)
    write_jsonl(CODEX / "analysis" / "v36_tool_skeleton_library.jsonl", skeletons)
    write_jsonl(CODEX / "analysis" / "v36_instantiated_trajectories.jsonl", [{k: v for k, v in r.items() if k not in {"item", "results"}} for r in candidate_rows])
    write_jsonl(CODEX / "analysis" / "v36_execution_trace.jsonl", [{k: v for k, v in r.items() if k != "item"} for r in candidate_rows])
    write_jsonl(CODEX / "analysis" / "v36_verifier_scores.jsonl", [{"task_key": r["task_key"], "candidate_id": r["candidate_id"], "verifier": r["verifier"], "score": r["score"], "program_family": r.get("program_family"), "source": r.get("source")} for r in candidate_rows])
    write_jsonl(CODEX / "analysis" / "v36_frozen_selection_trace.jsonl", selection_rows)
    write_jsonl(CODEX / "analysis" / "v36_oracle_best_trace.jsonl", oracle_rows)

    after_mtime = V10_ZIP.stat().st_mtime_ns if V10_ZIP.exists() else None
    state = {
        "run_id": run_id,
        "version": V36_VERSION,
        "preflight": {
            "v10_zip_exists": V10_ZIP.exists(),
            "v10_sha256": v10_sha,
            "v10_expected_sha256": V10_EXPECTED_SHA,
            "v10_zip_overwritten": before_mtime != after_mtime,
            "openai_env_exists": (CODEX / "state" / ".openai_env").exists(),
            "final_run": False,
            "uses_final_hidden_metadata": False,
        },
        "data_usage": {
            "uses_val41_gt_for_runtime": False,
            "uses_val41_gt_for_post_eval": True,
            "uses_final_hidden_metadata": False,
            "nonfinal_skeleton_count": len(skeletons),
        },
        "baseline": {"V34_selected": v34_summary},
        "protected_success_count": len(protected_keys),
        "target_task_count": len(target_keys),
        "candidate_count": len(candidate_rows),
        "lattice_task_count": len(lattice_rows),
        "frozen_selected": {"result_dir": str(frozen_dir), "summary": frozen_eval["summary"], "new_joint_vs_v34": frozen_new},
        "oracle_best": {"result_dir": str(oracle_dir), "summary": oracle_eval["summary"], "new_joint_vs_v34": oracle_new},
        "final_run": False,
        "uses_final_hidden_metadata": False,
        "auto_submit": False,
    }
    f_count = score_count(frozen_eval["summary"])
    o_count = score_count(oracle_eval["summary"])
    if f_count > 11:
        state["decision"] = "V36 frozen selected exceeds V34; consider this as next non-final val41 candidate."
    elif o_count > 11:
        state["decision"] = "Oracle-best exceeds V34 but frozen selector does not; repair verifier/selector before promotion."
    else:
        state["decision"] = "V36 skeleton/lattice candidate pool did not beat V34; do not promote."
    write_json(CODEX / "state" / "latest_v36_db_lattice.json", state)
    (CODEX / "state" / "latest_v36_run_id.txt").write_text(run_id + "\n", encoding="utf-8")
    write_reports(run_id, state, lattice_rows, skeletons, candidate_rows, selection_rows, oracle_rows)
    print(json.dumps(state, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
