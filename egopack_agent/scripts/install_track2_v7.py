#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Install Track2 V7 human-prior wrapper modules on the remote host."""

from __future__ import annotations

import hashlib
import os
import shutil
import textwrap
import time
from pathlib import Path


CODEX = Path("/home/data-gxu/acm/egolink2026-main/code/track2/codex")
WRAP = CODEX / "wrappers" / "egobench_agent_plus"
SCRIPTS = CODEX / "scripts"


MODULES = {}

MODULES["tool_affordance_memory.py"] = r'''
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
    for name in sorted(schema):
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
    elif scenario in {"retail", "restaurant"}:
        families = ["read_only_retrieval", "state_changing_add", "state_changing_remove", "aggregate_compute"]
    else:
        families = ["read_only_retrieval"]
    return {
        "allowed_families": families,
        "candidate_tools": tools_by_family(scenario, families, limit=5),
    }
'''

MODULES["human_process_graph.py"] = r'''
# -*- coding: utf-8 -*-
"""Human process graphs for Track2 scenarios.

The graph is a compact process-shape prior. It guides coverage of stages, not
task-specific answers.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from .tool_affordance_memory import describe_allowed_families, tool_family


GRAPH_TEMPLATES: Dict[str, List[Dict[str, Any]]] = {
    "retail": [
        {"stage": "identify_product_or_visible_candidate", "allowed_families": ["read_only_retrieval"], "risky_if_skipped": True},
        {"stage": "retrieve_product_attributes", "allowed_families": ["read_only_retrieval"], "risky_if_skipped": True},
        {"stage": "compare_constraints", "allowed_families": ["read_only_retrieval"], "risky_if_skipped": False},
        {"stage": "apply_cart_or_list_mutation", "allowed_families": ["state_changing_add", "state_changing_remove"], "risky_if_skipped": True},
        {"stage": "compute_total_or_nutrition", "allowed_families": ["aggregate_compute"], "risky_if_skipped": False},
        {"stage": "final_response", "allowed_families": [], "risky_if_skipped": False},
    ],
    "kitchen": [
        {"stage": "identify_current_recipe_or_visible_ingredients", "allowed_families": ["read_only_retrieval"], "risky_if_skipped": True},
        {"stage": "get_recipe_ingredients_once", "allowed_families": ["read_only_retrieval"], "risky_if_skipped": True},
        {"stage": "determine_branch_from_menu_fridge_recipe", "allowed_families": ["read_only_retrieval"], "risky_if_skipped": True},
        {"stage": "apply_menu_or_shopping_list_mutation", "allowed_families": ["state_changing_add", "state_changing_remove"], "risky_if_skipped": True},
        {"stage": "compute_total_nutritions", "allowed_families": ["aggregate_compute"], "risky_if_skipped": False},
        {"stage": "final_response", "allowed_families": [], "risky_if_skipped": False},
    ],
    "restaurant": [
        {"stage": "identify_dish_or_set_meal", "allowed_families": ["read_only_retrieval"], "risky_if_skipped": True},
        {"stage": "retrieve_menu_attributes", "allowed_families": ["read_only_retrieval"], "risky_if_skipped": True},
        {"stage": "apply_order_mutation", "allowed_families": ["state_changing_add", "state_changing_remove"], "risky_if_skipped": False},
        {"stage": "compute_total_nutrition_or_payment", "allowed_families": ["aggregate_compute"], "risky_if_skipped": False},
        {"stage": "final_response", "allowed_families": [], "risky_if_skipped": False},
    ],
    "order": [
        {"stage": "pin_restaurant", "allowed_families": ["read_only_retrieval"], "prerequisite_slots": ["restaurant_name"], "risky_if_skipped": True},
        {"stage": "inspect_current_order_or_menu_context", "allowed_families": ["read_only_retrieval"], "risky_if_skipped": True},
        {"stage": "identify_target_dish_or_set_meal", "allowed_families": ["read_only_retrieval"], "risky_if_skipped": True},
        {"stage": "add_new_dish_if_needed", "allowed_families": ["state_changing_add"], "risky_if_skipped": False},
        {"stage": "remove_old_dish_or_set_meal_if_needed", "allowed_families": ["state_changing_remove"], "risky_if_skipped": False},
        {"stage": "compute_tax_or_payment", "allowed_families": ["aggregate_compute"], "risky_if_skipped": True},
        {"stage": "final_response", "allowed_families": [], "risky_if_skipped": False},
    ],
}


def _calls(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    return list(state.get("executed_tool_calls") or [])


def _families(calls: List[Dict[str, Any]]) -> List[str]:
    return [tool_family(str(c.get("tool_name", ""))) for c in calls if isinstance(c, dict)]


def _tool_names(calls: List[Dict[str, Any]]) -> List[str]:
    return [str(c.get("tool_name", "")) for c in calls if isinstance(c, dict)]


def _text_needs_mutation(text: str) -> bool:
    t = str(text or "").lower()
    return bool(re.search(r"\b(add|remove|replace|delete|order|cart|shopping list|menu|include|exclude)\b", t))


def infer_process_state(scenario: str, state: Dict[str, Any]) -> Dict[str, Any]:
    scenario = str(scenario or state.get("scenario") or "")
    calls = _calls(state)
    fams = _families(calls)
    names = _tool_names(calls)
    pins = state.get("pins") or {}
    instruction = str(state.get("user_instruction") or "")
    stage = "final_response"
    missing: List[str] = []
    coverage: Dict[str, bool] = {
        "has_retrieval": "read_only_retrieval" in fams,
        "has_add": "state_changing_add" in fams,
        "has_remove": "state_changing_remove" in fams,
        "has_update": "state_changing_update" in fams,
        "has_aggregate": "aggregate_compute" in fams,
    }
    if scenario == "order":
        if not pins.get("restaurant_name"):
            stage = "pin_restaurant"
            missing.append("restaurant_name")
        elif not coverage["has_retrieval"]:
            stage = "inspect_current_order_or_menu_context"
        elif _text_needs_mutation(instruction) and not (coverage["has_add"] or coverage["has_remove"]):
            stage = "identify_target_dish_or_set_meal"
        elif coverage["has_add"] and not coverage["has_remove"] and any(w in instruction.lower() for w in ("remove", "replace")):
            stage = "remove_old_dish_or_set_meal_if_needed"
        elif (coverage["has_add"] or coverage["has_remove"] or state.get("order_requested_final_aggregate")) and not coverage["has_aggregate"]:
            stage = "compute_tax_or_payment"
            missing.append("final_aggregate")
        else:
            stage = "final_response"
    elif scenario == "kitchen":
        if not coverage["has_retrieval"]:
            stage = "identify_current_recipe_or_visible_ingredients"
        elif not any("get_recipe_ingredients" in n for n in names):
            stage = "get_recipe_ingredients_once"
        elif _text_needs_mutation(instruction) and not (coverage["has_add"] or coverage["has_remove"]):
            stage = "determine_branch_from_menu_fridge_recipe"
        elif (coverage["has_add"] or coverage["has_remove"]) and not coverage["has_aggregate"]:
            stage = "compute_total_nutritions"
            missing.append("final_nutrition_compute")
        else:
            stage = "final_response"
        if state.get("tool_call_count", 0) > 25:
            missing.append("STOP_EXPLORING")
    elif scenario == "retail":
        if not coverage["has_retrieval"]:
            stage = "identify_product_or_visible_candidate"
        elif _text_needs_mutation(instruction) and not (coverage["has_add"] or coverage["has_remove"]):
            stage = "apply_cart_or_list_mutation"
        elif (coverage["has_add"] or coverage["has_remove"]) and not coverage["has_aggregate"]:
            stage = "compute_total_or_nutrition"
        else:
            stage = "final_response"
    elif scenario == "restaurant":
        if not coverage["has_retrieval"]:
            stage = "identify_dish_or_set_meal"
        elif _text_needs_mutation(instruction) and not (coverage["has_add"] or coverage["has_remove"]):
            stage = "apply_order_mutation"
        else:
            stage = "final_response"
    allowed = describe_allowed_families(scenario, stage)
    return {
        "scenario": scenario,
        "current_stage": stage,
        "missing_prerequisites": missing,
        "allowed_tool_families": allowed.get("allowed_families", []),
        "allowed_tool_set": allowed.get("candidate_tools", []),
        "expected_next_tool_family": (allowed.get("allowed_families") or ["message"])[0],
        "process_coverage_state": coverage,
        "template_nodes": [n["stage"] for n in GRAPH_TEMPLATES.get(scenario, [])],
    }


def prompt_snippet(scenario: str, state: Dict[str, Any]) -> str:
    ps = infer_process_state(scenario, state)
    pins = state.get("pins") or {}
    lines = [
        "Human-prior process state:",
        f"- scenario: {scenario}",
        f"- current_stage: {ps['current_stage']}",
        f"- pinned_user_id: {pins.get('user_id') or ''}",
        f"- pinned_restaurant_name: {pins.get('restaurant_name') or ''}",
        f"- allowed_next_families: {', '.join(ps['allowed_tool_families']) or 'short_message'}",
        f"- candidate_tools_cap: {', '.join(ps['allowed_tool_set'][:5])}",
    ]
    if ps["missing_prerequisites"]:
        lines.append(f"- missing_prerequisites: {', '.join(ps['missing_prerequisites'])}")
    return "\n".join(lines)
'''

MODULES["process_coverage_verifier.py"] = r'''
# -*- coding: utf-8 -*-
"""Process coverage verifier for Track2 V7.

This verifier checks shape-level process coverage only. It does not read GT
answers and does not encode task-specific final parameters.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .tool_affordance_memory import tool_family
from .human_process_graph import infer_process_state


AGG_TOOLS = ("compute_total_tax", "compute_total_payment", "compute_total_price", "compute_total_nutrition", "compute_total_nutritions")


def _calls_from_state(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    return list(state.get("executed_tool_calls") or [])


def _names(calls: List[Dict[str, Any]]) -> List[str]:
    return [str(c.get("tool_name", "")) for c in calls if isinstance(c, dict)]


def _families(calls: List[Dict[str, Any]]) -> List[str]:
    return [tool_family(str(c.get("tool_name", ""))) for c in calls if isinstance(c, dict)]


def verify_process_coverage(
    scenario: str,
    candidate_calls: Any = None,
    state: Dict[str, Any] | None = None,
    natural_reply: bool = False,
) -> Dict[str, Any]:
    state = state or {}
    prior = _calls_from_state(state)
    cand = candidate_calls if isinstance(candidate_calls, list) else ([candidate_calls] if isinstance(candidate_calls, dict) else [])
    all_calls = prior + cand
    names = _names(all_calls)
    fams = _families(all_calls)
    ps = infer_process_state(scenario, state)
    score = 0.0
    missing: List[str] = []
    mismatch: List[str] = []
    should_continue = False
    should_stop = False
    suggested = ps.get("expected_next_tool_family", "read_only_retrieval")

    if "read_only_retrieval" in fams:
        score += 0.25
    if any(f.startswith("state_changing") for f in fams):
        score += 0.25
    if "aggregate_compute" in fams:
        score += 0.25
    if ps.get("current_stage") == "final_response":
        score += 0.25

    if scenario == "order":
        pins = state.get("pins") or {}
        if not pins.get("restaurant_name"):
            missing.append("pin_restaurant")
            suggested = "read_only_retrieval"
        has_mutation = any(f in {"state_changing_add", "state_changing_remove"} for f in fams)
        has_aggregate = any(str(n).startswith(AGG_TOOLS) for n in names)
        requested_agg = bool(state.get("order_requested_final_aggregate"))
        if (has_mutation or requested_agg) and not has_aggregate:
            missing.append("final_aggregate_after_order_process")
            suggested = "aggregate_compute"
            should_continue = True
        compute_payment_count = sum(1 for n in names if n == "compute_total_payment")
        if compute_payment_count >= 2 and "state_changing_remove" not in fams and any(w in str(state.get("user_instruction", "")).lower() for w in ("remove", "replace")):
            mismatch.append("repeated_payment_loop_before_remove")
            suggested = "state_changing_remove"
            should_continue = True
        for c in cand:
            if not isinstance(c, dict):
                continue
            name = str(c.get("tool_name", ""))
            params = c.get("parameters", {}) if isinstance(c.get("parameters", {}), dict) else {}
            if name == "remove_dish_from_order" and params.get("set_meal_name"):
                mismatch.append("set_meal_sent_to_dish_remove_tool")
            if name == "remove_set_meal_from_order" and params.get("dish_name") and not params.get("set_meal_name"):
                mismatch.append("dish_name_sent_to_set_meal_remove_tool")
    elif scenario == "kitchen":
        has_recipe = any("get_recipe_ingredients" in n for n in names)
        has_mutation = any(f in {"state_changing_add", "state_changing_remove"} for f in fams)
        has_compute = any(n.startswith("compute_total_nutritions") for n in names)
        if not has_recipe and any(w in str(state.get("user_instruction", "")).lower() for w in ("recipe", "menu", "shopping", "nutrition")):
            missing.append("recipe_ingredient_retrieval")
            suggested = "read_only_retrieval"
        if has_mutation and not has_compute and "nutrition" in str(state.get("user_instruction", "")).lower():
            missing.append("compute_total_nutritions")
            suggested = "aggregate_compute"
            should_continue = True
        if state.get("tool_call_count", 0) > 35:
            mismatch.append("kitchen_conservative_mode_active")
    elif scenario == "retail":
        if state.get("blocked_calls"):
            mismatch.append("duplicate_or_risky_mutation_seen")
    elif scenario == "restaurant":
        should_stop = natural_reply and not missing

    if not missing and not mismatch and ps.get("current_stage") == "final_response":
        should_stop = True
    return {
        "process_coverage_score": min(1.0, score),
        "missing_process_stage": missing,
        "suggested_next_tool_family": suggested,
        "should_continue": should_continue,
        "should_stop": should_stop,
        "should_retry": bool(mismatch),
        "tool_family_mismatch": mismatch,
        "process_state": ps,
    }
'''

MODULES["counterfactual_db_simulator.py"] = r'''
# -*- coding: utf-8 -*-
"""Counterfactual DB simulator lite for pre-execution sanity checks."""

from __future__ import annotations

import json
import re
import string
from typing import Any, Dict, List

from .tool_affordance_memory import tool_family


def _canon(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    table = str.maketrans("", "", "".join(ch for ch in string.punctuation if ch not in "&'"))
    return text.translate(table).strip()


def _entity(params: Dict[str, Any]) -> str:
    for key in ("product_name", "dish_name", "set_meal_name", "ingredient_name", "recipe_name", "category"):
        if params.get(key):
            return f"{key}:{_canon(params.get(key))}"
    return ""


def _mutation_key(tool_name: str, params: Dict[str, Any], scenario: str, state: Dict[str, Any]) -> str:
    pins = state.get("pins") or {}
    user = params.get("user_id") or params.get("customer_id") or pins.get("user_id") or ""
    rest = params.get("restaurant_name") or pins.get("restaurant_name") or ""
    return "|".join([tool_family(tool_name), scenario, _canon(user), _canon(rest), _entity(params)])


def assess_call(tool_name: str, params: Dict[str, Any], scenario: str, state: Dict[str, Any]) -> Dict[str, Any]:
    params = params if isinstance(params, dict) else {}
    family = tool_family(tool_name)
    pins = state.get("pins") or {}
    risk = 0.0
    reasons: List[str] = []
    action = "allow"
    repaired_params = None

    if scenario == "order" and family.startswith("state_changing"):
        if not (params.get("restaurant_name") or pins.get("restaurant_name")):
            risk += 0.6
            reasons.append("order_mutation_without_restaurant_pin")
            action = "require_retrieval"
    if family.startswith("state_changing"):
        key = _mutation_key(tool_name, params, scenario, state)
        for ledger_key in (state.get("successful_mutation_ledger") or {}):
            if key and key in str(ledger_key):
                risk += 0.8
                reasons.append("duplicate_state_change_counterfactual")
                action = "block"
                break
    if scenario == "order" and str(tool_name) == "remove_dish_from_order" and params.get("dish_name"):
        # Existing db_guard has the canonical set-meal rewrite. Here we only
        # expose the process risk for telemetry/verifier weights.
        name_l = str(params.get("dish_name", "")).lower()
        if any(piece in name_l for piece in ("set meal", "combo", "platter", "meal for")):
            risk += 0.35
            reasons.append("possible_set_meal_in_dish_remove")
    if scenario == "kitchen" and str(tool_name).startswith("compute_total_nutritions"):
        ingredients = params.get("ingredients")
        if not isinstance(ingredients, list) or not ingredients:
            risk += 0.7
            reasons.append("nutrition_compute_without_confirmed_ingredients")
            action = "require_retrieval"

    return {
        "action": action,
        "allow": action == "allow",
        "risk_score": round(min(1.0, risk), 3),
        "risk_reason": reasons,
        "repaired_params": repaired_params,
        "before_after_prediction": {
            "family": family,
            "entity": _entity(params),
            "ledger_size": len(state.get("successful_mutation_ledger") or {}),
        },
    }


def assess_batch(calls: Any, scenario: str, state: Dict[str, Any]) -> List[Dict[str, Any]]:
    call_list = calls if isinstance(calls, list) else ([calls] if isinstance(calls, dict) else [])
    out = []
    for call in call_list:
        if not isinstance(call, dict):
            continue
        out.append({
            "tool_name": call.get("tool_name"),
            "decision": assess_call(str(call.get("tool_name", "")), call.get("parameters", {}), scenario, state),
        })
    return out
'''

MODULES["visual_slot_prior.py"] = r'''
# -*- coding: utf-8 -*-
"""Selective visual-to-slot prior for V7."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List


CODEX_ROOT = Path(os.environ.get("CODEX_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/codex"))


def _first_json(path: Path) -> Dict[str, Any]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _cache_dirs(scenario: str, task_id: Any) -> List[Path]:
    root = CODEX_ROOT / "visual_cache"
    if not root.exists():
        return []
    return sorted(root.glob(f"{scenario}*_{task_id}"))


def load_visual_slots(scenario: str, state: Dict[str, Any], visual_context: Dict[str, Any] | None = None) -> Dict[str, Any]:
    visual_context = visual_context or {}
    task_id = state.get("task_id")
    cache_dir = None
    cache_id = visual_context.get("cache_id")
    if cache_id:
        c = CODEX_ROOT / "visual_cache" / str(cache_id)
        if c.exists():
            cache_dir = c
    if cache_dir is None:
        dirs = _cache_dirs(scenario, task_id)
        cache_dir = dirs[0] if dirs else None
    data: Dict[str, Any] = {}
    contact_sheet = ""
    if cache_dir:
        data = _first_json(cache_dir / "visual_state.json")
        if (cache_dir / "contact_sheet.jpg").exists():
            contact_sheet = str(cache_dir / "contact_sheet.jpg")
    text = visual_context.get("visual_state_text") or ""
    if not text and cache_dir and (cache_dir / "visual_state.txt").exists():
        try:
            text = (cache_dir / "visual_state.txt").read_text(encoding="utf-8")
        except Exception:
            text = ""
    blob = json.dumps(data, ensure_ascii=False) + "\n" + str(text)
    def pull(pattern: str, limit: int = 5) -> List[str]:
        out = []
        for m in re.findall(pattern, blob, flags=re.I):
            val = re.sub(r"\s+", " ", str(m)).strip(" -:;,.'\"")
            if val and val not in out:
                out.append(val)
            if len(out) >= limit:
                break
        return out
    slots = {
        "restaurant_name": (data.get("restaurant_name") or data.get("restaurant_name_candidates") or [None])[0] if isinstance(data.get("restaurant_name_candidates"), list) else data.get("restaurant_name"),
        "category_candidates": data.get("category_candidates") or pull(r"\b([A-Z][A-Za-z ]{2,30}(?:Pasta|Steaks|Desserts|Drinks|Salads|Meals))\b"),
        "dish_candidates": data.get("dish_candidates") or pull(r"\b([A-Z][A-Za-z' -]{2,40}(?:Chicken|Beef|Pork|Pasta|Rice|Salad|Soup|Steak|Burger|Pizza|Fish|Cutlet))\b"),
        "set_meal_candidates": data.get("set_meal_candidates") or pull(r"\b([A-Z][A-Za-z' -]{2,40}(?:Set Meal|Combo|Platter))\b"),
        "product_candidates": data.get("product_candidates") or [],
        "ingredient_candidates": data.get("ingredient_candidates") or pull(r"\b([a-zA-Z][A-Za-z -]{2,25}(?:flour|pork|chicken|egg|milk|rice|oil|salt|sugar|onion|garlic))\b"),
        "current_recipe_candidates": data.get("current_recipe_candidates") or data.get("recipe_candidates") or [],
        "pointed_entity": data.get("pointed_entity") or "",
        "visible_text": data.get("visible_text") or "",
        "action_sequence": data.get("temporal_events") or data.get("action_sequence") or "",
        "uncertainty": data.get("uncertainty_notes") or "",
        "contact_sheet_path": contact_sheet,
        "source": str(cache_dir) if cache_dir else "",
        "enabled_for_retry": bool(contact_sheet and scenario in {"order", "kitchen"}),
    }
    # Cap lists; visual slots are candidates only.
    for key, val in list(slots.items()):
        if isinstance(val, list):
            slots[key] = [str(x) for x in val if str(x).strip()][:5]
    return slots


def compact_slot_text(slots: Dict[str, Any], scenario: str) -> str:
    if not slots:
        return ""
    keys = ["restaurant_name", "category_candidates", "dish_candidates", "set_meal_candidates", "ingredient_candidates", "current_recipe_candidates", "pointed_entity", "uncertainty"]
    lines = ["Visual-to-slot prior candidates (verify via tools before mutation):"]
    for key in keys:
        val = slots.get(key)
        if val:
            lines.append(f"- {key}: {val}")
    if scenario == "order" and not slots.get("contact_sheet_path"):
        lines.append("- visual_retry: disabled; no contact_sheet for this task")
    return "\n".join(lines)
'''

MODULES["working_memory_manager.py"] = r'''
# -*- coding: utf-8 -*-
"""Working-memory cap for Track2 V7 prompts."""

from __future__ import annotations

import json
from typing import Any, Dict, List


def compact_history(history: List[Dict[str, Any]], limit: int = 3) -> List[Dict[str, str]]:
    out = []
    for msg in (history or [])[-limit:]:
        content = str(msg.get("content", ""))
        out.append({"role": str(msg.get("role", "")), "content": content[:700]})
    return out


def summarize_ledgers(state: Dict[str, Any]) -> Dict[str, Any]:
    ledger = state.get("successful_mutation_ledger") or {}
    blocked = state.get("blocked_calls") or []
    return {
        "successful_mutations": min(len(ledger), 20),
        "blocked_calls": len(blocked),
        "tool_call_count": state.get("tool_call_count", 0),
        "kitchen_stage": state.get("kitchen_stage"),
    }


def cap_list(values: Any, limit: int = 5) -> List[str]:
    if not isinstance(values, list):
        return [str(values)] if values else []
    out = []
    for value in values:
        text = str(value).strip()
        if text and text not in out:
            out.append(text)
        if len(out) >= limit:
            break
    return out


def build_working_memory_prompt(scenario: str, state: Dict[str, Any], history: List[Dict[str, Any]], process_state: Dict[str, Any], visual_slots: Dict[str, Any]) -> str:
    pins = state.get("pins") or {}
    top_slots = {}
    for key in ("restaurant_name", "dish_candidates", "set_meal_candidates", "ingredient_candidates", "product_candidates", "current_recipe_candidates", "pointed_entity"):
        val = visual_slots.get(key)
        if val:
            top_slots[key] = cap_list(val, 3) if isinstance(val, list) else str(val)[:200]
    prompt = {
        "current_scenario": scenario,
        "current_process_stage": process_state.get("current_stage"),
        "pinned": pins,
        "allowed_next_tool_families": process_state.get("allowed_tool_families", [])[:5],
        "active_candidate_tools": process_state.get("allowed_tool_set", [])[:5],
        "mutation_ledger_summary": summarize_ledgers(state),
        "top_visual_slots": top_slots,
        "recent_history": compact_history(history, 3),
    }
    if scenario == "kitchen" and state.get("tool_call_count", 0) > 25:
        prompt["mode"] = "STOP_EXPLORING"
    if scenario == "kitchen" and state.get("tool_call_count", 0) > 35:
        prompt["mode"] = "CONSERVATIVE_ONLY"
    return "Compact working memory:\n" + json.dumps(prompt, ensure_ascii=False, separators=(",", ":"))[:2600]
'''

MODULES["human_prior_controller.py"] = r'''
# -*- coding: utf-8 -*-
"""Controller glue for V7 human-prior modules."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

from .counterfactual_db_simulator import assess_batch
from .human_process_graph import infer_process_state, prompt_snippet
from .process_coverage_verifier import verify_process_coverage
from .visual_slot_prior import load_visual_slots, compact_slot_text
from .working_memory_manager import build_working_memory_prompt


CODEX_ROOT = Path(os.environ.get("CODEX_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/codex"))


def human_prior_enabled() -> bool:
    return os.environ.get("TRACK2_ENABLE_HUMAN_PRIOR", "0") == "1"


def level() -> str:
    forced = os.environ.get("TRACK2_HUMAN_PRIOR_LEVEL")
    if forced:
        return forced
    version = os.environ.get("TRACK2_RUN_VERSION", "")
    if "V7_4" in version:
        return "full"
    if "V7_3" in version:
        return "helpers"
    if "V7_2" in version:
        return "counterfactual"
    if "V7_1" in version:
        return "verifier"
    if "V7_0" in version:
        return "graph"
    return "full"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def event_path(state: Dict[str, Any]) -> Path:
    version = state.get("version") or os.environ.get("TRACK2_RUN_VERSION") or "V7"
    run_id = state.get("run_id") or os.environ.get("TRACK2_RUN_ID") or time.strftime("manual_%Y%m%d_%H%M%S")
    task_id = state.get("task_id", "unknown")
    path = CODEX_ROOT / "runs" / str(version) / str(run_id) / "human_prior_events" / f"{task_id}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def append_human_prior_event(state: Dict[str, Any] | None, event: Dict[str, Any]) -> None:
    if not state:
        return
    record = {
        "time": _now(),
        "task_id": state.get("task_id"),
        "scenario": state.get("scenario"),
        "version": state.get("version"),
        "run_id": state.get("run_id"),
        "human_prior_level": level(),
        **event,
    }
    try:
        with event_path(state).open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


def append_policy_trace(state: Dict[str, Any] | None, record: Dict[str, Any]) -> None:
    if not state:
        return
    try:
        path = CODEX_ROOT / "train_data" / "human_prior_policy_traces.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        out = {
            "time": _now(),
            "task_id": state.get("task_id"),
            "scenario": state.get("scenario"),
            "user_utterance": state.get("user_instruction"),
            "source_version": state.get("version"),
            **record,
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    except Exception:
        pass


STATIC_PROMPT = """
Human-prior service policy:
- Act like a careful service worker: confirm the goal, retrieve necessary evidence, mutate state once, then verify with the final aggregate tool.
- Cover process stages, not just final DB state. If an order mutation occurs, finish the requested tax/payment/nutrition aggregate before a final message.
- Never repeat a successful non-idempotent state-changing call.
- Do not treat set meals as ordinary dishes; use set_meal tools for set_meal_name and dish tools for dish_name.
- Do not modify an order before restaurant_name is pinned.
- For kitchen, get recipe ingredients once, choose the branch from evidence, avoid broad scans, and compute nutrition only from confirmed ingredients.
- Keep active tool/entity candidates small. If tools are needed, output only a JSON array; otherwise output only a short user-facing sentence.
""".strip()


def static_prompt() -> str:
    return STATIC_PROMPT if human_prior_enabled() else ""


def build_turn_system_message(scenario: str, state: Dict[str, Any], history: List[Dict[str, Any]], visual_context: Dict[str, Any] | None = None, turn: int | None = None) -> str:
    if not human_prior_enabled():
        return ""
    ps = infer_process_state(scenario, state)
    slots = load_visual_slots(scenario, state, visual_context)
    wm = build_working_memory_prompt(scenario, state, history, ps, slots)
    text = "\n\n".join([
        prompt_snippet(scenario, state),
        compact_slot_text(slots, scenario),
        wm,
    ]).strip()
    append_human_prior_event(state, {
        "event": "human_prior_turn_prompt",
        "turn": turn,
        "process_state": ps,
        "visual_slots": slots,
    })
    return text[:5000]


def observe_model_reply(reply: str, scenario: str, history: List[Dict[str, Any]], state: Dict[str, Any] | None, turn: int | None = None) -> None:
    if not human_prior_enabled() or not state:
        return
    verdict = verify_process_coverage(scenario, None, state, natural_reply=True)
    append_human_prior_event(state, {
        "event": "human_prior_model_reply",
        "turn": turn,
        "raw_model_output": str(reply)[:4000],
        "verifier_decision": verdict,
    })


def observe_validated_output(raw_reply: str, repaired_output: str, normalized_output: str, scenario: str, state: Dict[str, Any] | None, turn: int | None = None, validation: Dict[str, Any] | None = None) -> None:
    if not human_prior_enabled() or not state:
        return
    try:
        calls = json.loads(normalized_output)
    except Exception:
        calls = []
    verifier = verify_process_coverage(scenario, calls, state, natural_reply=False)
    counterfactual = assess_batch(calls, scenario, state) if level() in {"counterfactual", "helpers", "full"} else []
    process_state = infer_process_state(scenario, state)
    slots = load_visual_slots(scenario, state, None)
    append_human_prior_event(state, {
        "event": "human_prior_validated_action",
        "turn": turn,
        "process_stage": process_state.get("current_stage"),
        "raw_model_output": str(raw_reply)[:4000],
        "repaired_output": repaired_output,
        "validated_action": normalized_output,
        "verifier_decision": verifier,
        "counterfactual_decision": counterfactual,
        "validation": validation or {},
    })
    append_policy_trace(state, {
        "visual_slots": slots,
        "process_stage": process_state.get("current_stage"),
        "allowed_tools": process_state.get("allowed_tool_set", []),
        "forbidden_tools": [],
        "model_raw_output": str(raw_reply)[:4000],
        "repaired_output": repaired_output,
        "verifier_decision": verifier,
        "counterfactual_decision": counterfactual,
        "final_executed_action": normalized_output,
    })
'''


RUN_SCRIPT = r'''#!/usr/bin/env bash
set -euo pipefail

CODEX=/home/data-gxu/acm/egolink2026-main/code/track2/codex
EGO=/home/data-gxu/acm/egolink2026-main/code/track2/EgoBench
PYTHON_BIN="${TRACK2_PYTHON:-python3}"
RUN_ID="${TRACK2_RUN_ID:-human_prior_gate_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$CODEX"/{logs,reports,analysis,state,runs,train_data}
cd "$CODEX"

if [[ -f "$CODEX/state/.openai_env" ]]; then
  # shellcheck disable=SC1091
  source "$CODEX/state/.openai_env"
fi

MODEL="${TRACK2_OPENAI_MODEL:-gpt-5.5}"
BASE_URL="${TRACK2_OPENAI_BASE_URL:-https://ai-pixel.online/v1}"

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

export CODEX_ROOT="$CODEX"
export EGO_ROOT="$EGO"
export SERVICE_MODEL_BACKEND="${SERVICE_MODEL_BACKEND:-openai_compatible_chat}"
export SERVICE_MODEL_NAME="$MODEL"
export SERVICE_MODEL_API_BASE="$BASE_URL"
export SERVICE_MODEL_API_KEY="${OPENAI_API_KEY:-}"
export USER_AGENT_API_BASE_URL="$BASE_URL"
export USER_AGENT_API_KEY="${OPENAI_API_KEY:-}"
export USER_MODEL_NAME="$MODEL"
export TRACK2_USER_USE_OPENAI=0
export TRACK2_USE_OPENAI_GPT55=0
export TRACK2_GPT55_STRUCTURED_OUTPUT=0
export TRACK2_ENABLE_DB_GUARD=1
export TRACK2_ENABLE_PLANNER=1
export TRACK2_ENABLE_SCENARIO_RULES=1
export TRACK2_ENABLE_VISUAL_CACHE=1
export TRACK2_TEXT_ONLY_VISUAL_CONTEXT=1
export TRACK2_USE_VIDEO=0
export TRACK2_ENABLE_HUMAN_PRIOR=1
export TRACK2_MAX_TURNS="${TRACK2_MAX_TURNS:-6}"
export TRACK2_DEFAULT_MAX_TOKENS="${TRACK2_DEFAULT_MAX_TOKENS:-2048}"
export TRACK2_CONNECT_TIMEOUT="${TRACK2_CONNECT_TIMEOUT:-10}"
export TRACK2_READ_TIMEOUT="${TRACK2_READ_TIMEOUT:-240}"
export TRACK2_API_MAX_RETRIES="${TRACK2_API_MAX_RETRIES:-1}"
export TRACK2_TEMPERATURE="${TRACK2_TEMPERATURE:-0.1}"
export TRACK2_RUN_VERSION="${TRACK2_RUN_VERSION:-V7_4_human_prior_full}"
export TRACK2_RUN_ID="$RUN_ID"
export TRACK2_OUTPUT_MODEL_NAME="${MODEL}-${TRACK2_RUN_VERSION}-${RUN_ID}"
export PYTHONPATH="$CODEX/wrappers:$CODEX:${PYTHONPATH:-}"

case "$TRACK2_RUN_VERSION" in
  V7_0*) export TRACK2_HUMAN_PRIOR_LEVEL=graph ;;
  V7_1*) export TRACK2_HUMAN_PRIOR_LEVEL=verifier ;;
  V7_2*) export TRACK2_HUMAN_PRIOR_LEVEL=counterfactual ;;
  V7_3*) export TRACK2_HUMAN_PRIOR_LEVEL=helpers ;;
  *) export TRACK2_HUMAN_PRIOR_LEVEL=full ;;
esac

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  cat > "$CODEX/reports/HUMAN_PRIOR_GATE_SUMMARY_${RUN_ID}.md" <<MD
# Human Prior Gate Summary

- generated_at: $(date +%Y-%m-%dT%H:%M:%S%z)
- run_id: $RUN_ID
- version: $TRACK2_RUN_VERSION
- status: not_run_key_missing
- key_present: no
- final_auto_submitted: no
MD
  exit 2
fi

mkdir -p "$CODEX/runs/$TRACK2_RUN_VERSION/$RUN_ID/logs"
cd "$EGO"

GATE_SPECS="${TRACK2_GATE_SPECS:-retail:9 restaurant:4 order:1 kitchen:2}"
for spec in $GATE_SPECS; do
  scenario="${spec%%:*}"
  num="${spec##*:}"
  "$PYTHON_BIN" "$CODEX/runners/track2_multi_agent_plus.py" \
    --scenario "$scenario" --scenario_number "$num" \
    --service_model_name "$MODEL" --num_tasks 1 \
    > "$CODEX/runs/$TRACK2_RUN_VERSION/$RUN_ID/logs/${scenario}${num}.log" 2>&1 || true
done

cd "$EGO/analysis_scripts"
"$PYTHON_BIN" evaluate_interaction.py --model_name "$TRACK2_OUTPUT_MODEL_NAME" --num_samples 1 \
  > "$CODEX/runs/$TRACK2_RUN_VERSION/$RUN_ID/logs/eval_python3.log" 2>&1 || true

cd "$CODEX"
"$PYTHON_BIN" "$CODEX/scripts/track2_gpt55_collect_gate.py" \
  --run-id "$RUN_ID" --model "$MODEL" --version "$TRACK2_RUN_VERSION" --no-update-best || true

cat > "$CODEX/state/latest_human_prior_gate.json" <<JSON
{
  "updated_at": "$(date +%Y-%m-%dT%H:%M:%S%z)",
  "run_id": "$RUN_ID",
  "model": "$MODEL",
  "version": "$TRACK2_RUN_VERSION",
  "base_url": "$BASE_URL",
  "key_present": true,
  "key_logged": false,
  "report": "reports/02_gpt55_gate_summary_${RUN_ID}.md",
  "final_auto_submitted": false
}
JSON
'''


COLLECT_HUMAN_PRIOR = r'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collect V7 human-prior ablation runs and update best only under strict rules."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List


CODEX_ROOT = Path(os.environ.get("CODEX_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/codex"))
EGO_ROOT = Path(os.environ.get("EGO_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/EgoBench"))
TASKS = ["retail9", "restaurant4", "order1", "kitchen2"]


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def fval(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return 0.0


def metric(data: Dict[str, Any], dotted: str) -> Any:
    cur: Any = data
    for part in dotted.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def eval_metrics(output_model: str, task: str) -> Dict[str, Any]:
    path = EGO_ROOT / "eval_result" / output_model / f"{task}_easy_eval.json"
    data = load_json(path)
    if not isinstance(data, dict):
        return {"missing": True, "eval_path": str(path), "joint_success": 0.0, "result_success": 0.0, "tool_success": 0.0, "micro_tool_accuracy": 0.0, "avg_rounds": 0.0, "avg_tool_calls": 0.0, "tool_matches": 0.0, "gt_tool_calls": 0.0, "interaction_tool_calls": 0.0}
    detailed = data.get("detailed_results") or []
    tb = detailed[0].get("tool_based", {}) if detailed and isinstance(detailed[0], dict) else {}
    return {
        "missing": False,
        "eval_path": str(path),
        "joint_success": fval(metric(data, "joint_success.success_rate")),
        "result_success": fval(metric(data, "result_based.success_rate")),
        "tool_success": fval(metric(data, "tool_based.success_rate")),
        "micro_tool_accuracy": fval(metric(data, "micro_tool_stats.micro_accuracy")),
        "avg_rounds": fval(metric(data, "performance_metrics.avg_rounds_count")),
        "avg_tool_calls": fval(metric(data, "performance_metrics.avg_tool_calls_count")),
        "tool_matches": fval(tb.get("matches")),
        "gt_tool_calls": fval(tb.get("total_gt_calls")),
        "interaction_tool_calls": fval(tb.get("total_interaction_calls")),
    }


def mean(rows: List[Dict[str, Any]], key: str) -> float:
    return sum(fval(r.get(key)) for r in rows) / max(1, len(rows))


def count_events(version: str, run_id: str) -> Dict[str, int]:
    root = CODEX_ROOT / "runs" / version / run_id
    counts = {"human_prior_events": 0, "policy_traces_total": 0, "counterfactual_decisions": 0, "process_verifier_events": 0, "visual_slot_events": 0, "duplicate_mutation_blocks": 0}
    for path in (root / "human_prior_events").glob("*.jsonl"):
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                counts["human_prior_events"] += 1
                rec = json.loads(line)
                if rec.get("counterfactual_decision"):
                    counts["counterfactual_decisions"] += 1
                if rec.get("verifier_decision"):
                    counts["process_verifier_events"] += 1
                if rec.get("visual_slots"):
                    counts["visual_slot_events"] += 1
        except Exception:
            pass
    for path in (root / "wrapper_events").glob("*.jsonl"):
        try:
            counts["duplicate_mutation_blocks"] += path.read_text(encoding="utf-8").count("duplicate_mutation_blocked")
        except Exception:
            pass
    trace_path = CODEX_ROOT / "train_data" / "human_prior_policy_traces.jsonl"
    if trace_path.exists():
        try:
            counts["policy_traces_total"] = sum(1 for _ in trace_path.open("r", encoding="utf-8"))
        except Exception:
            pass
    return counts


def summarize_run(version: str, run_id: str, model: str) -> Dict[str, Any]:
    output_model = f"{model}-{version}-{run_id}"
    rows = []
    for task in TASKS:
        row = {"version": version, "run_id": run_id, "output_model": output_model, "task": task}
        row.update(eval_metrics(output_model, task))
        rows.append(row)
    summary = {
        "version": version,
        "run_id": run_id,
        "model": model,
        "output_model": output_model,
        "joint_success": mean(rows, "joint_success"),
        "result_success": mean(rows, "result_success"),
        "tool_success": mean(rows, "tool_success"),
        "micro_tool_accuracy": mean(rows, "micro_tool_accuracy"),
        "avg_rounds": mean(rows, "avg_rounds"),
        "avg_tool_calls": mean(rows, "avg_tool_calls"),
    }
    summary.update(count_events(version, run_id))
    return {"summary": summary, "rows": rows}


def should_update_best(candidate: Dict[str, Any], rows: List[Dict[str, Any]], best: Dict[str, Any]) -> bool:
    best_joint = fval(best.get("joint_success"))
    cand_joint = fval(candidate.get("joint_success"))
    cand_micro = fval(candidate.get("micro_tool_accuracy"))
    cand_result = fval(candidate.get("result_success"))
    if cand_joint > best_joint:
        return True
    retail_ok = next((r for r in rows if r["task"] == "retail9"), {}).get("joint_success", 0) >= 1.0
    restaurant_ok = next((r for r in rows if r["task"] == "restaurant4"), {}).get("joint_success", 0) >= 1.0
    order_micro = next((r for r in rows if r["task"] == "order1"), {}).get("micro_tool_accuracy", 0)
    if abs(cand_joint - best_joint) < 1e-9 and cand_micro > 0.7083 and cand_result >= 0.75:
        return True
    if abs(cand_joint - best_joint) < 1e-9 and order_micro > 0.3334 and retail_ok and restaurant_ok:
        return True
    return False


def write_reports(all_rows: List[Dict[str, Any]], summaries: List[Dict[str, Any]], updated: bool, best_candidate: Dict[str, Any] | None, stamp: str) -> None:
    analysis = CODEX_ROOT / "analysis" / f"human_prior_ablation_{stamp}.csv"
    analysis.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted(set().union(*(r.keys() for r in all_rows))) if all_rows else ["version"]
    with analysis.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)

    impl = CODEX_ROOT / "reports" / f"HUMAN_PRIOR_IMPLEMENTATION_{stamp}.md"
    impl.write_text("\n".join([
        "# Human Prior Implementation",
        "",
        f"- generated_at: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}",
        "- base_version: V6_1_3_gpt55_guarded_endpoint",
        "- final_auto_submitted: no",
        "- api_key_logged: no",
        "",
        "## Modules",
        "",
        "- Human Process Graph: stage prior for retail/kitchen/restaurant/order.",
        "- Tool Affordance Memory: schema-derived read/mutate/aggregate tags.",
        "- Process-Coverage Verifier: shape checks for order aggregate and kitchen branch flow.",
        "- Counterfactual DB Simulator Lite: pre-execution risk checks from pins and mutation ledger.",
        "- Visual-to-Slot Prior: cached visual_state candidates only, verified by tools before mutation.",
        "- Working Memory Manager: caps prompt state to pins, current stage, ledgers, recent turns, and top slots.",
        "- Human Prior Controller: telemetry and policy trace glue behind TRACK2_ENABLE_HUMAN_PRIOR=1.",
        "",
        f"- ablation_csv: {analysis}",
    ]) + "\n", encoding="utf-8")

    gate = CODEX_ROOT / "reports" / f"HUMAN_PRIOR_GATE_SUMMARY_{stamp}.md"
    lines = [
        "# Human Prior Gate Summary",
        "",
        f"- generated_at: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}",
        "- final_auto_submitted: no",
        "- api_key_logged: no",
        f"- best_state_updated: {'yes' if updated else 'no'}",
        f"- selected_candidate: {(best_candidate or {}).get('version', 'none')}",
        "",
        "| version | run_id | joint | result | tool | micro | avg_tool_calls | hp_events | cf_decisions | duplicate_blocks |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        lines.append(f"| {s['version']} | {s['run_id']} | {s['joint_success']:.3f} | {s['result_success']:.3f} | {s['tool_success']:.3f} | {s['micro_tool_accuracy']:.3f} | {s['avg_tool_calls']:.2f} | {s.get('human_prior_events',0)} | {s.get('counterfactual_decisions',0)} | {s.get('duplicate_mutation_blocks',0)} |")
    lines += ["", "## Per-Task Rows", "", "| version | task | joint | result | tool | micro | tool_calls | matches/gt |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in all_rows:
        lines.append(f"| {r['version']} | {r['task']} | {r['joint_success']:.3f} | {r['result_success']:.3f} | {r['tool_success']:.3f} | {r['micro_tool_accuracy']:.3f} | {r['avg_tool_calls']:.1f} | {r['tool_matches']:.0f}/{r['gt_tool_calls']:.0f} |")
    gate.write_text("\n".join(lines) + "\n", encoding="utf-8")

    ablation = CODEX_ROOT / "reports" / f"HUMAN_PRIOR_ABLATION_{stamp}.md"
    ablation.write_text("\n".join([
        "# Human Prior Ablation",
        "",
        f"- generated_at: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}",
        "- baseline: V6_1_3_gpt55_guarded_endpoint joint 0.50, result 0.75, tool 0.50, micro 0.7083",
        "- full_run_expanded: no",
        "- reason: only expand if 4-task joint >= 0.75 or order/kitchen clearly improves.",
        "",
        "## Summary",
        "",
        "\n".join(f"- {s['version']}: joint={s['joint_success']:.3f}, result={s['result_success']:.3f}, tool={s['tool_success']:.3f}, micro={s['micro_tool_accuracy']:.3f}, avg_tool_calls={s['avg_tool_calls']:.2f}" for s in summaries),
    ]) + "\n", encoding="utf-8")

    paper = CODEX_ROOT / "reports" / f"HUMAN_PRIOR_PAPER_NOTES_{stamp}.md"
    paper.write_text("\n".join([
        "# Human-Prior Tool Agent for Egocentric Service Tasks",
        "",
        f"- generated_at: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}",
        "- Track2 policy: commercial API allowed; GPT-5.5 used as service agent through OpenAI-compatible endpoint.",
        "- final_auto_submitted: no",
        "- claim boundary: motivated by human cognitive priors; effectiveness must be judged by dev gate/ablation metrics.",
        "",
        "## Components",
        "",
        "- Human Process Graph: encodes scenario process stages without GT answers.",
        "- Tool Affordance Memory: derives tool risk/timing tags from schema.",
        "- Visual-to-Slot Prior: converts cached visual evidence into candidate slots, never direct final answers.",
        "- Counterfactual DB Simulator: checks mutation consequences against pins and episode ledger.",
        "- Process-Coverage Verifier: targets result/tool mismatch by requiring aggregate stages after mutations.",
        "- Working Memory Manager: caps active tool and entity candidates to reduce order/kitchen drift.",
        "- Socially Robust User Guidance: short replies and no misleading follow-up after subgoals complete.",
        "",
        "## Ablation Metrics",
        "",
        "\n".join(f"- {s['version']}: joint={s['joint_success']:.3f}, micro={s['micro_tool_accuracy']:.3f}, events={s.get('human_prior_events',0)}" for s in summaries),
    ]) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("TRACK2_OPENAI_MODEL", "gpt-5.5"))
    ap.add_argument("--runs", nargs="+", required=True, help="version:run_id pairs")
    args = ap.parse_args()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    all_rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    expanded = []
    for item in args.runs:
        version, run_id = item.split(":", 1)
        got = summarize_run(version, run_id, args.model)
        summaries.append(got["summary"])
        all_rows.extend(got["rows"])
        expanded.append(got)
    best_path = CODEX_ROOT / "state" / "best_track2_api_version.json"
    current_best = load_json(best_path) or {}
    updated = False
    best_candidate = None
    for got in expanded:
        if should_update_best(got["summary"], got["rows"], current_best):
            if best_candidate is None or (got["summary"]["joint_success"], got["summary"]["micro_tool_accuracy"]) > (best_candidate["joint_success"], best_candidate["micro_tool_accuracy"]):
                best_candidate = got["summary"]
    if best_candidate:
        out = {
            "version": best_candidate["version"],
            "joint_success": best_candidate["joint_success"],
            "tool_success": best_candidate["tool_success"],
            "result_success": best_candidate["result_success"],
            "micro_tool_accuracy": best_candidate["micro_tool_accuracy"],
            "avg_rounds": best_candidate["avg_rounds"],
            "avg_tool_calls": best_candidate["avg_tool_calls"],
            "run_id": best_candidate["run_id"],
            "model": best_candidate["model"],
            "endpoint": os.environ.get("TRACK2_OPENAI_BASE_URL", "https://ai-pixel.online/v1"),
            "external_api_used": True,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "note": "Updated by V7 human-prior strict gate criterion; final not auto-submitted.",
        }
        best_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        updated = True
    write_reports(all_rows, summaries, updated, best_candidate, stamp)
    state = CODEX_ROOT / "state" / "latest_human_prior_ablation.json"
    state.write_text(json.dumps({"updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "runs": args.runs, "best_updated": updated, "selected": best_candidate}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"reports_stamp": stamp, "best_updated": updated, "selected": best_candidate}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_if_changed(path: Path, text: str) -> bool:
    text = textwrap.dedent(text).lstrip("\n")
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return True


def backup(paths: list[Path]) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    dst = CODEX / "backups" / f"v7_human_prior_{stamp}"
    dst.mkdir(parents=True, exist_ok=True)
    for path in paths:
        if not path.exists():
            continue
        rel = path.relative_to(CODEX)
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    return dst


def patch_file(path: Path, replacements: list[tuple[str, str]]) -> bool:
    text = path.read_text(encoding="utf-8")
    original = text
    for old, new in replacements:
        if old not in text:
            continue
        text = text.replace(old, new, 1)
    if text != original:
        path.write_text(text, encoding="utf-8")
        return True
    return False


def install() -> None:
    CODEX.mkdir(parents=True, exist_ok=True)
    targets = [
        CODEX / "state" / "best_track2_api_version.json",
        WRAP / "service_agent_wrapper.py",
        WRAP / "prompt_builder.py",
        WRAP / "db_guard.py",
        CODEX / "runners" / "track2_multi_agent_plus.py",
    ]
    bdir = backup(targets)
    changed = []
    for name, content in MODULES.items():
        path = WRAP / name
        if write_if_changed(path, content):
            changed.append(str(path))
    run_path = SCRIPTS / "run_human_prior_gate.sh"
    if write_if_changed(run_path, RUN_SCRIPT):
        changed.append(str(run_path))
    os.chmod(run_path, 0o755)
    collect_path = SCRIPTS / "track2_collect_human_prior.py"
    if write_if_changed(collect_path, COLLECT_HUMAN_PRIOR):
        changed.append(str(collect_path))
    os.chmod(collect_path, 0o755)

    # prompt_builder: add static human-prior prompt behind env flag.
    pb = WRAP / "prompt_builder.py"
    if "from .human_prior_controller import static_prompt as human_prior_static_prompt" not in pb.read_text(encoding="utf-8"):
        patch_file(pb, [
            ("from .planner import planner_prompt\n", "from .planner import planner_prompt\ntry:\n    from .human_prior_controller import static_prompt as human_prior_static_prompt\nexcept Exception:\n    def human_prior_static_prompt():\n        return \"\"\n"),
            ("    if os.environ.get(\"TRACK2_ENABLE_PLANNER\") == \"1\":\n", "    hp_prompt = human_prior_static_prompt()\n    if hp_prompt:\n        text += \"\\n\\n\" + hp_prompt\n    if os.environ.get(\"TRACK2_ENABLE_PLANNER\") == \"1\":\n"),
        ])
        changed.append(str(pb))

    # service_agent_wrapper: observe raw and validated outputs.
    sw = WRAP / "service_agent_wrapper.py"
    if "observe_model_reply" not in sw.read_text(encoding="utf-8"):
        patch_file(sw, [
            ("from .tool_validator import validate_tool_json\n", "from .tool_validator import validate_tool_json\ntry:\n    from .human_prior_controller import observe_model_reply, observe_validated_output\nexcept Exception:\n    def observe_model_reply(*args, **kwargs):\n        return None\n    def observe_validated_output(*args, **kwargs):\n        return None\n"),
            ("    if not _looks_like_tool_attempt(reply):\n", "    observe_model_reply(reply, scenario, history, episode_state, turn)\n    if not _looks_like_tool_attempt(reply):\n"),
            ("        record[\"guard\"] = guard\n", "        record[\"guard\"] = guard\n        observe_validated_output(reply, repaired, normalized, scenario, episode_state, turn, validation)\n"),
            ("            record[\"final_action\"] = normalized\n", "            observe_validated_output(reply, repaired, normalized, scenario, episode_state, turn, validation)\n            record[\"final_action\"] = normalized\n"),
        ])
        changed.append(str(sw))

    # db_guard: initialize executed calls and record post execution; add human-prior counterfactual telemetry.
    dg = WRAP / "db_guard.py"
    text_dg = dg.read_text(encoding="utf-8")
    repls = []
    if "\"executed_tool_calls\": []," not in text_dg:
        repls.append(("        \"external_api_used\": False,\n", "        \"external_api_used\": False,\n        \"executed_tool_calls\": [],\n"))
    if "human_prior_counterfactual" not in text_dg:
        repls.append(("        if scenario == \"kitchen\":\n", "        if os.environ.get(\"TRACK2_ENABLE_HUMAN_PRIOR\", \"0\") == \"1\":\n            try:\n                from .counterfactual_db_simulator import assess_call\n                from .human_prior_controller import append_human_prior_event\n                cf = assess_call(name, params, scenario, state)\n                append_human_prior_event(state, {\"event\": \"human_prior_counterfactual\", \"turn\": turn, \"tool_name\": name, \"parameters\": params, \"decision\": cf})\n                if cf.get(\"action\") == \"block\":\n                    content = \"Counterfactual DB simulator blocked risky state-changing call: \" + \", \".join(cf.get(\"risk_reason\") or [])\n                    synthetic_results.append(_synthetic_result(content, call2))\n                    decisions.append({\"tool_name\": name, \"decision\": \"block\", \"reason\": \"human_prior_counterfactual\", \"counterfactual\": cf})\n                    continue\n            except Exception:\n                pass\n\n        if scenario == \"kitchen\":\n"))
    if "state.setdefault(\"executed_tool_calls\", []).append" not in text_dg:
        repls.append(("        if is_final_compute_tool(name) and _looks_success(result):\n", "        state.setdefault(\"executed_tool_calls\", []).append({\"turn\": turn, \"tool_name\": name, \"parameters\": params, \"success\": _looks_success(result), \"result_preview\": str(result)[:500]})\n        if is_final_compute_tool(name) and _looks_success(result):\n"))
    if repls:
        patch_file(dg, repls)
        changed.append(str(dg))

    # runner: add dynamic turn-level human-prior prompt.
    runner = CODEX / "runners" / "track2_multi_agent_plus.py"
    rt = runner.read_text(encoding="utf-8")
    if "build_human_prior_turn_message" not in rt:
        patch_file(runner, [
            ("from egobench_agent_plus.direct_api import call_llm_direct\n", "from egobench_agent_plus.direct_api import call_llm_direct\ntry:\n    from egobench_agent_plus.human_prior_controller import build_turn_system_message as build_human_prior_turn_message\nexcept Exception:\n    def build_human_prior_turn_message(*args, **kwargs):\n        return \"\"\n"),
            ("                    for i, msg in enumerate(local_service_history):\n", "                    hp_turn_prompt = build_human_prior_turn_message(args.scenario, episode_state, local_service_history, visual_context, turn)\n                    if hp_turn_prompt:\n                        current_service_msgs.append({\"role\": \"system\", \"content\": hp_turn_prompt})\n                    for i, msg in enumerate(local_service_history):\n"),
        ])
        changed.append(str(runner))

    patch_dir = CODEX / "patches"
    patch_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "backup_dir": str(bdir),
        "changed": changed,
        "sha256": {str(p): sha256(Path(p)) for p in changed if Path(p).exists()},
        "note": "V7 human-prior wrapper modules; no official EgoBench source modified.",
    }
    (patch_dir / f"v7_human_prior_manifest_{stamp}.json").write_text(
        __import__("json").dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(__import__("json").dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    install()
