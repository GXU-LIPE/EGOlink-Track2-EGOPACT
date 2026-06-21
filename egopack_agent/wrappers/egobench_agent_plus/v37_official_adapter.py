#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V37 adapter for the freshly synchronized official EgoBench tree.

This module is intentionally read-only with respect to EgoBench.  It discovers
current tools, DB init objects, materialized val41 tasks, and safe query calls
without relying on stale hard-coded tool schemas.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


CODEX = Path(os.environ.get("CODEX_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/codex"))
EGO = Path(os.environ.get("EGO_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/EgoBench"))
SPLIT_DIR = CODEX / "state" / "materialized_splits" / "validation_A_limit30"

for p in (CODEX / "wrappers", CODEX, EGO):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)


MUTATION_RE = re.compile(r"^(add|remove|delete|update|modify|place|set|clear|replace)_|_(to|from)_(cart|order|shopping_list|menu)$", re.I)
QUERY_PREFIXES = ("get_", "find_", "filter_", "list_", "search_", "query_", "compute_", "calculate_", "check_")


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


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    tmp.replace(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tool_name(tool: Dict[str, Any]) -> str:
    if not isinstance(tool, dict):
        return ""
    fn = tool.get("function")
    if isinstance(fn, dict) and (fn.get("name") or fn.get("tool_name")):
        return str(fn.get("name") or fn.get("tool_name"))
    return str(tool.get("name") or tool.get("tool_name") or "")


def tool_parameters(tool: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(tool, dict):
        return {}
    fn = tool.get("function")
    if isinstance(fn, dict) and isinstance(fn.get("parameters"), dict):
        return fn.get("parameters") or {}
    for key in ("parameters", "input_schema"):
        if isinstance(tool.get(key), dict):
            return tool.get(key) or {}
    return {}


def tool_required_params(tool: Dict[str, Any]) -> List[str]:
    params = tool_parameters(tool)
    req = params.get("required")
    if isinstance(req, list):
        return [str(x) for x in req]
    props = params.get("properties")
    if isinstance(props, dict):
        return list(props)
    return []


def load_tool_schema(scenario: str) -> List[Dict[str, Any]]:
    path = EGO / "tools" / scenario / f"{scenario}_tools.json"
    data = read_json(path, [])
    return data if isinstance(data, list) else []


def tool_schema_summary(scenario: str) -> Dict[str, Any]:
    rows = load_tool_schema(scenario)
    names = [tool_name(t) for t in rows if tool_name(t)]
    return {
        "path": str(EGO / "tools" / scenario / f"{scenario}_tools.json"),
        "count": len(names),
        "names": names,
        "params": {tool_name(t): tool_required_params(t) for t in rows if tool_name(t)},
    }


def scenario_from_spec(spec: str) -> Tuple[str, int]:
    m = re.match(r"^([a-zA-Z_]+)(\d+)$", spec)
    if not m:
        raise ValueError(f"bad spec: {spec}")
    return m.group(1), int(m.group(2))


def init_db(scenario: str, number: int) -> Any:
    if scenario == "retail":
        db_mod = importlib.import_module("tools.retail.retail_db")
        init_mod = importlib.import_module("tools.retail.retail_init")
        db = db_mod.RetailDB()
        data = getattr(init_mod, f"retail_init_data{number}")
        db.init_from_json(data)
        return db
    if scenario == "restaurant":
        db_mod = importlib.import_module("tools.restaurant.restaurant_db")
        init_mod = importlib.import_module("tools.restaurant.restaurant_init")
        db = db_mod.RestaurantDB()
        data = getattr(init_mod, f"restaurant_init_data{number}", None) or getattr(init_mod, "restaurant_init_data")
        db.init_from_json(data)
        return db
    if scenario == "order":
        db_mod = importlib.import_module("tools.order.order_db")
        init_mod = importlib.import_module("tools.order.order_init")
        db = db_mod.OrderDB()
        data = getattr(init_mod, f"order_init_data{number}", None) or getattr(init_mod, "order_init_data")
        db.init_from_json(data)
        return db
    if scenario == "kitchen":
        db_mod = importlib.import_module("tools.kitchen.kitchen_db")
        init_mod = importlib.import_module("tools.kitchen.kitchen_init")
        db = db_mod.KitchenDB()
        data = getattr(init_mod, f"kitchen_init_data{number}", None) or getattr(init_mod, "kitchen_init_data")
        db.init_from_json(data)
        return db
    raise ValueError(f"unknown scenario: {scenario}")


def public_db_methods(db: Any) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for name in dir(db):
        if name.startswith("_"):
            continue
        attr = getattr(db, name, None)
        if not callable(attr):
            continue
        try:
            sig = inspect.signature(attr)
            out[name] = [p for p in sig.parameters if p != "self"]
        except Exception:
            out[name] = []
    return out


def _obj_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    return dict(getattr(obj, "__dict__", {}) or {})


def db_entity_counts_and_samples(scenario: str, db: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {"counts": {}, "samples": {}, "users": []}
    if scenario == "retail":
        catalog = getattr(db, "catalog", {}) or {}
        carts = getattr(db, "user_carts", {}) or {}
        lists = getattr(db, "user_shopping_lists", {}) or {}
        out["counts"] = {"products": len(catalog), "user_carts": len(carts), "user_shopping_lists": len(lists)}
        out["samples"]["products"] = [str(_obj_dict(v).get("name") or k) for k, v in list(catalog.items())[:8]]
        out["samples"]["categories"] = list(dict.fromkeys(str(_obj_dict(v).get("category")) for v in catalog.values() if _obj_dict(v).get("category")))[:8]
        out["users"] = list(dict.fromkeys(list(carts) + list(lists)))[:10]
    elif scenario == "restaurant":
        catalog = getattr(db, "catalog", {}) or {}
        meals = getattr(db, "set_meals", {}) or {}
        orders = getattr(db, "user_orders", {}) or {}
        out["counts"] = {"dishes": len(catalog), "set_meals": len(meals), "user_orders": len(orders)}
        out["samples"]["dishes"] = [str(_obj_dict(v).get("name") or k) for k, v in list(catalog.items())[:8]]
        out["samples"]["set_meals"] = [str(_obj_dict(v).get("set_meal_name") or _obj_dict(v).get("name") or k) for k, v in list(meals.items())[:8]]
        out["samples"]["categories"] = list(dict.fromkeys(str(_obj_dict(v).get("category")) for v in catalog.values() if _obj_dict(v).get("category")))[:8]
        out["users"] = list(orders)[:10]
    elif scenario == "order":
        rests = getattr(db, "restaurants", {}) or {}
        dishes = 0
        meals = 0
        orders = 0
        dish_samples: List[str] = []
        meal_samples: List[str] = []
        users: List[str] = []
        for rname, store in rests.items():
            cat = store.get("catalog", {}) if isinstance(store, dict) else {}
            sms = store.get("set_meals", {}) if isinstance(store, dict) else {}
            uo = store.get("user_orders", {}) if isinstance(store, dict) else {}
            dishes += len(cat)
            meals += len(sms)
            orders += len(uo)
            dish_samples.extend(str(_obj_dict(v).get("name") or k) for k, v in list(cat.items())[:4])
            meal_samples.extend(str(_obj_dict(v).get("set_meal_name") or _obj_dict(v).get("name") or k) for k, v in list(sms.items())[:4])
            users.extend(list(uo)[:4])
        out["counts"] = {"restaurants": len(rests), "dishes": dishes, "set_meals": meals, "user_orders": orders}
        out["samples"]["restaurants"] = list(rests)[:8]
        out["samples"]["dishes"] = dish_samples[:8]
        out["samples"]["set_meals"] = meal_samples[:8]
        out["users"] = list(dict.fromkeys(users))[:10]
    elif scenario == "kitchen":
        ingredients = getattr(db, "ingredients", {}) or {}
        recipes = getattr(db, "recipes", {}) or {}
        menus = getattr(db, "user_menus", {}) or {}
        lists = getattr(db, "user_shopping_lists", {}) or {}
        out["counts"] = {"ingredients": len(ingredients), "recipes": len(recipes), "user_menus": len(menus), "user_shopping_lists": len(lists)}
        out["samples"]["ingredients"] = [str(_obj_dict(v).get("name") or k) for k, v in list(ingredients.items())[:8]]
        out["samples"]["recipes"] = [str(_obj_dict(v).get("name") or k) for k, v in list(recipes.items())[:8]]
        out["samples"]["categories"] = list(dict.fromkeys(str(_obj_dict(v).get("category")) for v in ingredients.values() if _obj_dict(v).get("category")))[:8]
        out["users"] = list(dict.fromkeys(list(menus) + list(lists)))[:10]
    return out


def canonical_values_for_params(scenario: str, db: Any) -> Dict[str, Any]:
    info = db_entity_counts_and_samples(scenario, db)
    samples = info.get("samples", {})
    users = info.get("users", []) or ["user_1"]
    values: Dict[str, Any] = {
        "user_id": users[0],
        "quantity": 1,
        "new_quantity": 1,
        "min_price": 0,
        "max_price": 9999,
        "price": 0,
        "tax_rate": 0,
    }
    if samples.get("products"):
        values["product_name"] = samples["products"][0]
        values["product_names"] = [samples["products"][0]]
    if samples.get("dishes"):
        values["dish_name"] = samples["dishes"][0]
        values["dish_names"] = [samples["dishes"][0]]
    if samples.get("set_meals"):
        values["set_meal_name"] = samples["set_meals"][0]
        values["set_meal_names"] = [samples["set_meals"][0]]
    if samples.get("restaurants"):
        values["restaurant_name"] = samples["restaurants"][0]
    if samples.get("ingredients"):
        values["ingredient_name"] = samples["ingredients"][0]
        values["ingredient_names"] = [samples["ingredients"][0]]
    if samples.get("recipes"):
        values["recipe_name"] = samples["recipes"][0]
        values["recipe_names"] = [samples["recipes"][0]]
    if samples.get("categories"):
        values["category"] = samples["categories"][0]
        values["category_name"] = samples["categories"][0]
    return values


def fill_params_for_method(db: Any, method_name: str, values: Dict[str, Any]) -> Tuple[bool, Dict[str, Any], List[str]]:
    if not hasattr(db, method_name):
        return False, {}, ["method_missing"]
    method = getattr(db, method_name)
    try:
        sig = inspect.signature(method)
    except Exception as exc:
        return False, {}, [f"signature_error:{type(exc).__name__}"]
    params: Dict[str, Any] = {}
    missing: List[str] = []
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if name in values:
            params[name] = values[name]
        elif name.endswith("_name") and name.replace("_name", "") in values:
            params[name] = values[name.replace("_name", "")]
        elif param.default is inspect._empty:
            missing.append(name)
    return not missing, params, missing


def generate_harmless_query_calls(scenario: str, db: Any, limit: int = 2) -> List[Dict[str, Any]]:
    tools = load_tool_schema(scenario)
    values = canonical_values_for_params(scenario, db)
    calls: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for tool in tools:
        name = tool_name(tool)
        if not name or name in seen or MUTATION_RE.search(name):
            continue
        if not name.startswith(QUERY_PREFIXES):
            continue
        ok, params, missing = fill_params_for_method(db, name, values)
        if not ok:
            continue
        calls.append({"tool_name": name, "parameters": params})
        seen.add(name)
        if len(calls) >= limit:
            break
    return calls


def execute_tool_call(db: Any, call: Dict[str, Any]) -> Dict[str, Any]:
    name = call.get("tool_name") or call.get("name")
    params = call.get("parameters") or {}
    if not name or not hasattr(db, name):
        return {"tool_name": name, "parameters": params, "status": "error", "result": f"missing tool: {name}"}
    ok, filtered, missing = fill_params_for_method(db, name, {**canonical_values_for_params("", db), **params})
    if not ok:
        filtered = params
    try:
        result = getattr(db, name)(**filtered)
        status = "error" if isinstance(result, dict) and result.get("status") == "error" else "success"
        return {"tool_name": name, "parameters": filtered, "status": status, "result": result}
    except Exception as exc:
        return {"tool_name": name, "parameters": filtered, "status": "error", "result": f"{type(exc).__name__}: {exc}"}


def load_val41_manifest() -> Dict[str, Any]:
    return read_json(SPLIT_DIR / "manifest.json", {})


def load_val41_tasks() -> List[Dict[str, Any]]:
    manifest = load_val41_manifest()
    out: List[Dict[str, Any]] = []
    for scenario, number, _indices in manifest.get("specs", []):
        spec = f"{scenario}{int(number)}"
        rows = read_json(SPLIT_DIR / f"{spec}.json", [])
        if not isinstance(rows, list):
            continue
        for pos, row in enumerate(rows):
            out.append(
                {
                    "scenario": str(scenario),
                    "number": int(number),
                    "spec": spec,
                    "local_pos": pos,
                    "index": int(row.get("_v8_original_index", pos)),
                    "row": row,
                }
            )
    return out


def select_one_per_scenario(tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    preferred = {"retail": "retail2::5", "restaurant": "restaurant4::6", "kitchen": "kitchen1::31", "order": "order1::3"}
    by_key = {f"{t['spec']}::{t['index']}": t for t in tasks}
    for scenario, key in preferred.items():
        if key in by_key:
            out[scenario] = by_key[key]
    for t in tasks:
        out.setdefault(t["scenario"], t)
    return [out[k] for k in sorted(out)]


def select_mini_shadow_tasks(tasks: List[Dict[str, Any]], limit: int = 12) -> List[Dict[str, Any]]:
    preferred = [
        "kitchen1::31",
        "kitchen3::12",
        "restaurant4::6",
        "retail2::5",
        "order1::3",
        "order1::17",
        "restaurant3::24",
        "restaurant3::54",
        "retail4::13",
        "retail1::24",
        "kitchen2::33",
        "retail8::13",
    ]
    by_key = {f"{t['spec']}::{t['index']}": t for t in tasks}
    selected: List[Dict[str, Any]] = [by_key[k] for k in preferred if k in by_key]
    for t in tasks:
        if len(selected) >= limit:
            break
        key = f"{t['spec']}::{t['index']}"
        if key not in preferred:
            selected.append(t)
    return selected[:limit]


def safe_runtime_row(row: Dict[str, Any], idx: int) -> Dict[str, Any]:
    return {
        "Instruction": row.get("Instruction", ""),
        "image_path": row.get("image_path", ""),
        "task_id": row.get("task_id", 1),
        "_v8_original_index": idx,
    }
