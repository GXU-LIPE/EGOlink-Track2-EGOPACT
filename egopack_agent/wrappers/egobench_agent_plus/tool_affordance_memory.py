# -*- coding: utf-8 -*-
"""Tool affordance tags for Track2 human-prior control.

The module reads the existing EgoBench schema cache and derives conservative
tool tags. It describes process shape and risk; it never encodes dev answers.
"""

from __future__ import annotations

import json
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

from .schema_loader import load_schema, get_scenario_schema


CODEX_ROOT = Path(os.environ.get("CODEX_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/codex"))

READ_PREFIXES = ("get_", "find_", "search_", "list_", "check_", "query_", "retrieve_")
AGG_PREFIXES = (
    "compute_total_nutrition",
    "compute_total_nutritions",
    "compute_total_tax",
    "compute_total_price",
    "compute_total_payment",
)
ADD_MARKERS = ("add", "_to_cart", "_to_order", "_to_shopping_list", "_to_menu")
REMOVE_MARKERS = ("remove", "delete", "_from_cart", "_from_order", "_from_shopping_list", "_from_menu")
UPDATE_MARKERS = ("update", "modify", "set_")
ENTITY_FIELDS = ("product_name", "dish_name", "set_meal_name", "ingredient_name", "recipe_name", "category")


def _name(entry_or_name: Any) -> str:
    if isinstance(entry_or_name, dict):
        return str(entry_or_name.get("name") or "")
    return str(entry_or_name or "")


def _required(entry: Dict[str, Any]) -> Set[str]:
    return {str(x) for x in entry.get("required", []) or []}


def tool_family(tool_name: str) -> str:
    name = str(tool_name or "").lower()
    if name.startswith(AGG_PREFIXES):
        return "aggregate_compute"
    if name.startswith("remove") or "_from_" in name or name.startswith("delete"):
        return "state_changing_remove"
    if name.startswith("add") or "_to_" in name:
        return "state_changing_add"
    if name.startswith("update") or name.startswith("modify") or name.startswith("set_"):
        return "state_changing_update"
    if name.startswith(READ_PREFIXES) or "summary" in name or "current" in name:
        return "read_only_retrieval"
    return "other"


def tags_for_tool(entry_or_name: Any) -> List[str]:
    entry = entry_or_name if isinstance(entry_or_name, dict) else {}
    name = _name(entry_or_name).lower()
    req = _required(entry)
    family = tool_family(name)
    tags: Set[str] = {family}
    if family == "read_only_retrieval":
        tags.add("idempotent")
    if family.startswith("state_changing"):
        tags.update({"risky", "non_idempotent", "requires_prior_retrieval"})
    if family == "aggregate_compute":
        tags.update({"final_only", "requires_prior_retrieval"})
    if "user_id" in req or "customer_id" in req or "user_id" in name:
        tags.add("requires_user_id")
    if "restaurant_name" in req or "restaurant" in name:
        tags.add("requires_restaurant_name")
    if any(field in req for field in ENTITY_FIELDS) or any(piece in name for piece in ("dish", "meal", "product", "ingredient", "recipe", "category")):
        tags.add("requires_canonical_entity")
    if any(piece in name for piece in ("order", "cart", "shopping_list", "menu")) and family.startswith("state_changing"):
        tags.add("stateful_collection_mutation")
    return sorted(tags)


@lru_cache(maxsize=1)
def build_affordance_memory() -> Dict[str, Any]:
    schema = load_schema()
    memory: Dict[str, Any] = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "tools": {}, "scenarios": {}}
    for name, entry in (schema.get("tools") or {}).items():
        tags = tags_for_tool(entry)
        memory["tools"][name] = {
            "name": name,
            "scenario": entry.get("scenario"),
            "family": tool_family(name),
            "tags": tags,
            "required": entry.get("required", []),
            "optional": entry.get("optional", []),
        }
        memory["scenarios"].setdefault(entry.get("scenario", ""), []).append(name)
    try:
        out = CODEX_ROOT / "state" / "tool_affordance_memory.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(memory, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass
    return memory


def affordance_for_tool(tool_name: str) -> Dict[str, Any]:
    return (build_affordance_memory().get("tools") or {}).get(str(tool_name), {
        "name": str(tool_name),
        "family": tool_family(str(tool_name)),
        "tags": tags_for_tool(str(tool_name)),
    })


def tools_by_family(scenario: str, families: Iterable[str], limit: int = 8) -> List[str]:
    schema = get_scenario_schema(scenario)
    wanted = set(families)
    out: List[str] = []
    disallowed_management = {"add_product", "delete_product", "update_product", "add_dish", "delete_dish", "update_dish", "add_set_meal", "delete_set_meal", "update_set_meal"}
    for name in sorted(schema):
        if name in disallowed_management:
            continue
        fam = affordance_for_tool(name).get("family")
        if fam in wanted:
            out.append(name)
        if len(out) >= limit:
            break
    return out


def describe_allowed_families(scenario: str, stage: str) -> Dict[str, Any]:
    stage_l = str(stage or "").lower()
    if scenario == "order":
        if "pin" in stage_l or "inspect" in stage_l or "identify" in stage_l:
            families = ["read_only_retrieval"]
        elif "add" in stage_l:
            families = ["read_only_retrieval", "state_changing_add"]
        elif "remove" in stage_l:
            families = ["read_only_retrieval", "state_changing_remove"]
        elif "compute" in stage_l:
            families = ["aggregate_compute"]
        else:
            families = ["read_only_retrieval", "aggregate_compute"]
    elif scenario == "kitchen":
        if "compute" in stage_l:
            families = ["aggregate_compute", "read_only_retrieval"]
        elif "apply" in stage_l:
            families = ["state_changing_add", "state_changing_remove", "read_only_retrieval"]
        else:
            families = ["read_only_retrieval"]
    elif scenario == "retail":
        if "identify" in stage_l or "retrieve" in stage_l or "compare" in stage_l:
            families = ["read_only_retrieval"]
        elif "apply" in stage_l:
            families = ["read_only_retrieval", "state_changing_add", "state_changing_remove"]
        elif "compute" in stage_l:
            families = ["aggregate_compute", "read_only_retrieval"]
        else:
            families = ["read_only_retrieval"]
    elif scenario == "restaurant":
        if "identify" in stage_l or "retrieve" in stage_l:
            families = ["read_only_retrieval"]
        elif "apply" in stage_l:
            families = ["read_only_retrieval", "state_changing_add", "state_changing_remove"]
        elif "compute" in stage_l:
            families = ["aggregate_compute", "read_only_retrieval"]
        else:
            families = ["read_only_retrieval"]
    else:
        families = ["read_only_retrieval"]
    return {
        "allowed_families": families,
        "candidate_tools": tools_by_family(scenario, families, limit=5),
    }
