#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Install V8 helper modules and wire them into the existing guard."""
from pathlib import Path
import time

CODEX = Path('/home/data-gxu/acm/egolink2026-main/code/track2/codex')
WRAP = CODEX / 'wrappers' / 'egobench_agent_plus'
REPORTS = CODEX / 'reports'
REPORTS.mkdir(parents=True, exist_ok=True)

files = {}
files['v8_event_logger.py'] = r'''# -*- coding: utf-8 -*-
"""V8 event logger for Track2 helper modules."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict

CODEX_ROOT = Path(os.environ.get("CODEX_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/codex"))

def enabled(name: str) -> bool:
    return os.environ.get(name, "0") == "1"

def _safe(value: Any) -> Any:
    if isinstance(value, str):
        # Redact obvious API-key-looking strings without logging secrets.
        if value.startswith("sk-") and len(value) > 12:
            return "sk-[REDACTED]"
        return value[:4000]
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items() if "key" not in str(k).lower() and "token" not in str(k).lower()}
    if isinstance(value, list):
        return [_safe(v) for v in value[:50]]
    return value

def write_v8_event(state: Dict[str, Any] | None, module: str, decision: str, reason: str = "", **kwargs: Any) -> None:
    state = state or {}
    record = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "task_id": state.get("task_id"),
        "scenario": state.get("scenario"),
        "turn": kwargs.pop("turn", None),
        "module": module,
        "decision": decision,
        "reason": reason,
        "before_action": _safe(kwargs.pop("before_action", None)),
        "after_action": _safe(kwargs.pop("after_action", None)),
        "risk_score": kwargs.pop("risk_score", 0.0),
        "whether_repaired": kwargs.pop("whether_repaired", False),
        "whether_blocked": kwargs.pop("whether_blocked", False),
        "whether_crosschecked": kwargs.pop("whether_crosschecked", False),
        "whether_final_eval": os.environ.get("TRACK2_FINAL_EVAL", "0") == "1",
        "no_key_logged": True,
    }
    record.update({k: _safe(v) for k, v in kwargs.items()})
    version = state.get("version") or os.environ.get("TRACK2_RUN_VERSION", "V8")
    run_id = state.get("run_id") or os.environ.get("TRACK2_RUN_ID", "manual")
    task_id = state.get("task_id", "unknown")
    out = CODEX_ROOT / "runs" / str(version) / str(run_id) / "v8_events" / f"{task_id}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
'''

files['order_process_state_helper.py'] = r'''# -*- coding: utf-8 -*-
"""Deterministic V8 order process helper.

This module enforces process shape and type safety. It does not encode dev/final
answers and does not read final scenario JSON.
"""
from __future__ import annotations

import copy
import json
import re
from typing import Any, Dict, List, Tuple

from .db_guard import canonical_text, is_final_compute_tool, is_state_changing_tool
from .v8_event_logger import enabled, write_v8_event

SET_MEAL_HINTS = ("set meal", "combo", "platter", "meal for", "bundle")
AGG_TOOLS = {"compute_total_tax", "compute_total_payment", "compute_total_price"}
VISUAL_QUESTION_RE = re.compile(r"\b(what|which).{0,40}(see|shown|visible|menu|dish|category|picture|video)\b", re.I)

class OrderReplaceStateMachine:
    stages = [
        "O0_pin_restaurant",
        "O1_inspect_current_order_or_menu",
        "O2_identify_target_dish_or_set_meal",
        "O3_add_target_dish_if_needed",
        "O4_remove_old_dish_or_set_meal_if_needed",
        "O5_compute_tax_or_payment",
        "O6_final_response",
    ]

def _calls(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    return list(state.get("executed_tool_calls") or [])

def _names(state: Dict[str, Any]) -> List[str]:
    return [str(c.get("tool_name", "")) for c in _calls(state)]

def _intent(text: str) -> bool:
    t = str(text or "").lower()
    return any(w in t for w in ("replace", "remove", "change", "swap", "cancel", "set meal", "dish", "tax", "payment", "order"))

def infer_order_stage(state: Dict[str, Any]) -> str:
    pins = state.get("pins") or {}
    names = _names(state)
    if not pins.get("restaurant_name"):
        return "O0_pin_restaurant"
    if not any("order" in n.lower() or "dish" in n.lower() or "set_meal" in n.lower() or "set meal" in n.lower() for n in names):
        return "O1_inspect_current_order_or_menu"
    has_add = any(n.lower().startswith("add") and "order" in n.lower() for n in names)
    has_remove = any((n.lower().startswith("remove") or "from_order" in n.lower()) for n in names)
    if not (has_add or has_remove):
        return "O2_identify_target_dish_or_set_meal"
    if has_add and not has_remove and any(w in str(state.get("user_instruction", "")).lower() for w in ("replace", "remove", "swap", "change")):
        return "O4_remove_old_dish_or_set_meal_if_needed"
    if not any(n in AGG_TOOLS or n.startswith("compute_total") for n in names):
        return "O5_compute_tax_or_payment"
    return "O6_final_response"

def _normalize_aggregate_params(params: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    out = copy.deepcopy(params)
    changed = False
    dishes = out.get("dishes")
    if isinstance(dishes, list):
        for item in dishes:
            if isinstance(item, dict) and "product_name" in item and "dish_name" not in item:
                item["dish_name"] = item.pop("product_name")
                changed = True
    return out, changed

def _looks_set_meal(value: Any) -> bool:
    t = canonical_text(value)
    return any(h.replace(" ", "") in t.replace(" ", "") for h in SET_MEAL_HINTS)

def apply_order_helper(calls_obj: Any, state: Dict[str, Any], turn: int) -> Tuple[Any, List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not enabled("TRACK2_ENABLE_ORDER_HELPER") or state.get("scenario") != "order":
        return calls_obj, [], []
    calls = calls_obj if isinstance(calls_obj, list) else [calls_obj]
    out: List[Dict[str, Any]] = []
    synthetic: List[Dict[str, Any]] = []
    decisions: List[Dict[str, Any]] = []
    stage = infer_order_stage(state)
    state["order_v8_stage"] = stage
    write_v8_event(state, "order_helper", "stage", "order_stage_transition", turn=turn, risk_score=0.0, order_stage=stage)
    pins = state.get("pins") or {}
    for call in calls:
        if not isinstance(call, dict):
            continue
        call2 = copy.deepcopy(call)
        name = str(call2.get("tool_name", ""))
        params = call2.get("parameters", {})
        if not isinstance(params, dict):
            params = {}; call2["parameters"] = params
        lname = name.lower()
        before = copy.deepcopy(call2)
        if (is_state_changing_tool(name, "order") or is_final_compute_tool(name)) and pins.get("restaurant_name") and not params.get("restaurant_name"):
            params["restaurant_name"] = pins["restaurant_name"]
            decisions.append({"tool_name": name, "decision": "autofill_restaurant"})
            write_v8_event(state, "order_helper", "repair", "restaurant_pin_autofill", turn=turn, before_action=before, after_action=call2, whether_repaired=True)
        if (is_state_changing_tool(name, "order") or is_final_compute_tool(name)) and not (params.get("restaurant_name") or pins.get("restaurant_name")):
            synthetic.append({"role": "tool", "content": "Order helper blocked mutation/aggregate: pin restaurant_name via retrieval or dialogue before modifying/computing order.", "blocked": True, "tool_name": name})
            decisions.append({"tool_name": name, "decision": "block", "reason": "missing_restaurant_pin"})
            write_v8_event(state, "order_helper", "block", "missing_restaurant_pin", turn=turn, before_action=before, whether_blocked=True, risk_score=0.7)
            continue
        if name == "remove_dish_from_order" and (_looks_set_meal(params.get("dish_name")) or params.get("set_meal_name")):
            meal = params.get("set_meal_name") or params.get("dish_name")
            params.pop("dish_name", None); params["set_meal_name"] = meal; call2["tool_name"] = "remove_set_meal_from_order"; name = call2["tool_name"]
            decisions.append({"tool_name": "remove_dish_from_order", "decision": "rewrite", "to": name})
            write_v8_event(state, "order_helper", "repair", "order_setmeal_dish_type_rewrite", turn=turn, before_action=before, after_action=call2, whether_repaired=True, risk_score=0.35)
        if name in AGG_TOOLS or is_final_compute_tool(name):
            params2, changed = _normalize_aggregate_params(params)
            if changed:
                call2["parameters"] = params2; params = params2
                decisions.append({"tool_name": name, "decision": "rewrite", "reason": "aggregate_product_name_to_dish_name"})
                write_v8_event(state, "order_helper", "repair", "aggregate_dishes_use_dish_name", turn=turn, before_action=before, after_action=call2, whether_repaired=True)
            fp = json.dumps({"tool_name": name, "parameters": params}, ensure_ascii=False, sort_keys=True)
            if fp in state.setdefault("v8_order_compute_ledger", {}):
                synthetic.append({"role": "tool", "content": "Order aggregate loop blocked: same aggregate parameters were already computed. Return to missing add/remove stage or finish.", "blocked": True, "tool_name": name})
                decisions.append({"tool_name": name, "decision": "block", "reason": "aggregate_loop"})
                write_v8_event(state, "order_helper", "block", "order_aggregate_loop_blocked", turn=turn, before_action=before, whether_blocked=True, risk_score=0.6)
                continue
            state["v8_order_compute_ledger"][fp] = {"turn": turn, "tool_name": name}
            write_v8_event(state, "order_helper", "allow", "aggregate_tool_selected", turn=turn, after_action=call2)
        out.append(call2)
    return (out if isinstance(calls_obj, list) else (out[0] if out else [])), synthetic, decisions

def inspect_natural_reply(reply: str, state: Dict[str, Any], turn: int) -> Dict[str, Any]:
    if state.get("scenario") != "order" or not enabled("TRACK2_ENABLE_ORDER_HELPER"):
        return {"allow": True}
    no_visual = not state.get("contact_sheet_path") and os.environ.get("TRACK2_FINAL_EVAL", "0") == "1"
    if no_visual and VISUAL_QUESTION_RE.search(str(reply or "")):
        write_v8_event(state, "order_helper", "block", "order_no_visual_followup_blocked", turn=turn, before_action=reply, whether_blocked=True, risk_score=0.5)
        return {"allow": False, "replacement": "I will verify the order using the available restaurant/order tools instead of asking for visual details."}
    if _intent(state.get("user_instruction", "")) and infer_order_stage(state) != "O6_final_response":
        write_v8_event(state, "order_helper", "continue", "order_missing_stage_redirect", turn=turn, before_action=reply, order_stage=infer_order_stage(state), risk_score=0.4)
        return {"allow": False, "replacement": "I need to complete the required order tool process before finalizing."}
    return {"allow": True}
'''

files['kitchen_branch_helper.py'] = r'''# -*- coding: utf-8 -*-
"""Deterministic V8 kitchen branch helper."""
from __future__ import annotations

import copy
import json
from typing import Any, Dict, List, Tuple

from .v8_event_logger import enabled, write_v8_event

READ_ONLY = {"get_recipe_ingredients", "get_ingredient_location", "get_ingredient_quantity", "get_ingredient_nutrition", "find_ingredient_category", "find_ingredients_by_location", "get_current_menu", "get_current_shopping_list"}
BRANCH_CRITICAL = {"get_ingredient_quantity", "get_current_menu", "get_current_shopping_list", "get_recipe_ingredients"}
MUTATION = {"add_to_shopping_list", "add_recipe_to_menu", "remove_from_shopping_list", "remove_recipe_from_menu"}
FINAL = {"compute_total_nutritions", "compute_total_nutrition"}

def _key(tool_name: str, params: Dict[str, Any]) -> str:
    return json.dumps({"tool_name": tool_name, "parameters": params}, ensure_ascii=False, sort_keys=True)

def infer_kitchen_stage(state: Dict[str, Any]) -> str:
    names = [str(c.get("tool_name", "")) for c in state.get("executed_tool_calls") or []]
    if not any(n == "get_recipe_ingredients" for n in names):
        return "K0_identify_visible_or_current_recipe"
    if not any(n in BRANCH_CRITICAL for n in names):
        return "K2_intersect_current_menu_fridge_stock"
    if not any(n in MUTATION for n in names):
        return "K3_determine_missing_or_replacement_branch"
    if "nutrition" in str(state.get("user_instruction", "")).lower() and not any(n in FINAL for n in names):
        return "K5_compute_total_nutritions"
    return "K6_final_response"

def _is_broad_scan(tool_name: str, params: Dict[str, Any], state: Dict[str, Any]) -> bool:
    lname = tool_name.lower()
    if lname in {"find_ingredient_category", "get_ingredient_location"} and state.get("tool_call_count", 0) > 20:
        return True
    if lname == "get_recipe_ingredients" and len(state.setdefault("v8_kitchen_recipe_seen", [])) >= 2:
        recipe = str(params.get("recipe_name", "")).strip().lower()
        return recipe not in state.get("v8_kitchen_recipe_seen", [])
    return False

def _quantity_allowed(tool_name: str, params: Dict[str, Any], state: Dict[str, Any]) -> bool:
    if tool_name not in BRANCH_CRITICAL:
        return False
    count = state.setdefault("v8_kitchen_branch_quantity_count", 0)
    if count >= 3:
        return False
    text = " ".join([str(state.get("user_instruction", "")), json.dumps(params, ensure_ascii=False)]).lower()
    return any(w in text for w in ("recipe", "ingredient", "menu", "shopping", "fridge", "stock", "quantity", "nutrition"))

def apply_kitchen_helper(calls_obj: Any, state: Dict[str, Any], turn: int) -> Tuple[Any, List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not enabled("TRACK2_ENABLE_KITCHEN_HELPER") or state.get("scenario") != "kitchen":
        return calls_obj, [], []
    calls = calls_obj if isinstance(calls_obj, list) else [calls_obj]
    out: List[Dict[str, Any]] = []
    synthetic: List[Dict[str, Any]] = []
    decisions: List[Dict[str, Any]] = []
    stage = infer_kitchen_stage(state)
    state["kitchen_v8_stage"] = stage
    write_v8_event(state, "kitchen_helper", "stage", "kitchen_stage_transition", turn=turn, kitchen_stage=stage)
    for call in calls:
        if not isinstance(call, dict):
            continue
        call2 = copy.deepcopy(call)
        name = str(call2.get("tool_name", ""))
        lname = name.lower()
        params = call2.get("parameters", {})
        if not isinstance(params, dict):
            params = {}; call2["parameters"] = params
        before = copy.deepcopy(call2)
        if name == "get_recipe_ingredients":
            recipe = str(params.get("recipe_name", "")).strip().lower()
            seen = state.setdefault("v8_kitchen_recipe_seen", [])
            if recipe and recipe in seen:
                synthetic.append({"role": "tool", "content": "Kitchen helper skipped duplicate get_recipe_ingredients; use cached recipe ingredients.", "blocked": True, "tool_name": name})
                decisions.append({"tool_name": name, "decision": "block", "reason": "duplicate_recipe_ingredients"})
                write_v8_event(state, "kitchen_helper", "block", "duplicate_recipe_ingredients", turn=turn, before_action=before, whether_blocked=True, risk_score=0.3)
                continue
            if recipe:
                seen.append(recipe)
        qsig = _key(name, params)
        if name in READ_ONLY:
            cache = state.setdefault("v8_kitchen_readonly_cache", [])
            if qsig in cache:
                synthetic.append({"role": "tool", "content": "Kitchen helper skipped duplicate read-only query; use prior observation.", "blocked": True, "tool_name": name})
                decisions.append({"tool_name": name, "decision": "block", "reason": "duplicate_readonly_query"})
                write_v8_event(state, "kitchen_helper", "block", "duplicate_readonly_query", turn=turn, before_action=before, whether_blocked=True, risk_score=0.25)
                continue
            cache.append(qsig)
        tool_count = int(state.get("tool_call_count", 0))
        if tool_count > 35 and name not in MUTATION and name not in FINAL:
            if _quantity_allowed(name, params, state):
                state["v8_kitchen_branch_quantity_count"] += 1
                decisions.append({"tool_name": name, "decision": "allow", "reason": "branch_critical_quantity"})
                write_v8_event(state, "kitchen_helper", "allow", "kitchen_branch_critical_query_allowed", turn=turn, after_action=call2, risk_score=0.2)
            else:
                synthetic.append({"role": "tool", "content": "Kitchen helper blocked broad exploration in branch-aware conservative mode. Only branch-critical quantity, pending mutations, or final nutrition compute are allowed.", "blocked": True, "tool_name": name})
                decisions.append({"tool_name": name, "decision": "block", "reason": "branch_aware_conservative_mode"})
                write_v8_event(state, "kitchen_helper", "block", "kitchen_broad_scan_blocked", turn=turn, before_action=before, whether_blocked=True, risk_score=0.65)
                continue
        elif _is_broad_scan(name, params, state):
            synthetic.append({"role": "tool", "content": "Kitchen helper blocked broad scan. Stay on current recipe/menu/fridge branch and query only necessary quantities.", "blocked": True, "tool_name": name})
            decisions.append({"tool_name": name, "decision": "block", "reason": "broad_scan"})
            write_v8_event(state, "kitchen_helper", "block", "kitchen_broad_scan_blocked", turn=turn, before_action=before, whether_blocked=True, risk_score=0.5)
            continue
        if name in FINAL:
            ingredients = params.get("ingredients")
            if not isinstance(ingredients, list) or not ingredients:
                synthetic.append({"role": "tool", "content": "Kitchen helper blocked nutrition compute without ingredient provenance. Retrieve recipe/current menu/list quantities first.", "blocked": True, "tool_name": name})
                decisions.append({"tool_name": name, "decision": "block", "reason": "missing_nutrition_provenance"})
                write_v8_event(state, "kitchen_helper", "block", "kitchen_quantity_provenance", turn=turn, before_action=before, whether_blocked=True, risk_score=0.7)
                continue
            write_v8_event(state, "kitchen_helper", "allow", "kitchen_compute_ready", turn=turn, after_action=call2)
        out.append(call2)
    return (out if isinstance(calls_obj, list) else (out[0] if out else [])), synthetic, decisions
'''

files['deepseek_cross_validator.py'] = r'''# -*- coding: utf-8 -*-
"""Low-frequency DeepSeek cross-validator placeholder for V8.

The network call is intentionally off by default. This module provides trigger,
cache, and telemetry plumbing without replacing GPT-5.5 execution.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict

from .v8_event_logger import enabled, write_v8_event

CODEX_ROOT = Path(os.environ.get("CODEX_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/codex"))

def _cache_key(payload: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()

def should_crosscheck(scenario: str, state: Dict[str, Any], call: Any, risk_score: float = 0.0) -> bool:
    if not enabled("TRACK2_ENABLE_DEEPSEEK_CROSSCHECK"):
        return False
    if scenario not in {"order", "kitchen"}:
        return False
    if os.environ.get("TRACK2_USE_DEEPSEEK_CROSSCHECK", "0") != "1":
        return False
    if state.setdefault("v8_deepseek_crosscheck_count", 0) >= 2:
        return False
    return risk_score >= 0.4 or bool(state.get("blocked_calls"))

def crosscheck(payload: Dict[str, Any], state: Dict[str, Any], turn: int) -> Dict[str, Any]:
    cache_dir = CODEX_ROOT / "teacher_cache" / "deepseek_crosscheck"
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(payload)
    path = cache_dir / f"{key}.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        write_v8_event(state, "deepseek_crosscheck", "cache_hit", "deepseek_crosscheck_cache_hit", turn=turn, whether_crosschecked=True)
        return data
    # No online call unless the user explicitly enables TRACK2_USE_DEEPSEEK_CROSSCHECK=1
    # and later wiring provides a safe client. Return neutral decision now.
    data = {"risk": "low", "process_missing": [], "tool_type_confusion": [], "db_state_risk": [], "visual_grounding_risk": [], "recommended_action": "accept", "repair_hint": "", "confidence": 0.0, "online_call_performed": False}
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    state["v8_deepseek_crosscheck_count"] = state.get("v8_deepseek_crosscheck_count", 0) + 1
    write_v8_event(state, "deepseek_crosscheck", "accept", "deepseek_crosscheck_stub_no_online_call", turn=turn, whether_crosschecked=True)
    return data
'''

files['multicandidate_reranker.py'] = r'''# -*- coding: utf-8 -*-
"""Deterministic V8 candidate reranker."""
from __future__ import annotations

import json
from typing import Any, Dict, List

from .process_coverage_verifier import verify_process_coverage
from .counterfactual_db_simulator import assess_batch
from .v8_event_logger import enabled, write_v8_event


def score_candidate(candidate: Any, scenario: str, state: Dict[str, Any]) -> Dict[str, Any]:
    coverage = verify_process_coverage(scenario, candidate, state)
    cf = assess_batch(candidate, scenario, state)
    risk = max([float(x.get("decision", {}).get("risk_score", 0.0)) for x in cf] or [0.0])
    score = float(coverage.get("process_coverage_score", 0.0)) - risk
    if coverage.get("missing_process_stage"):
        score -= 0.2
    if coverage.get("tool_family_mismatch"):
        score -= 0.2
    return {"score": round(score, 4), "coverage": coverage, "counterfactual": cf, "risk": risk}

def select_candidate(candidates: List[Any], scenario: str, state: Dict[str, Any], turn: int) -> Dict[str, Any]:
    if not enabled("TRACK2_ENABLE_MULTICANDIDATE") or len(candidates) <= 1:
        return {"selected_index": 0, "selected": candidates[0] if candidates else None, "scores": []}
    scores = [score_candidate(c, scenario, state) for c in candidates]
    idx = max(range(len(scores)), key=lambda i: scores[i]["score"])
    write_v8_event(state, "multicandidate_reranker", "select", "multicandidate_score", turn=turn, selected_candidate=idx, scores=scores)
    return {"selected_index": idx, "selected": candidates[idx], "scores": scores, "rejected_candidates": [i for i in range(len(candidates)) if i != idx]}
'''

# write modules
for name, content in files.items():
    (WRAP / name).write_text(content, encoding='utf-8')

# Patch db_guard imports and calls.
db = WRAP / 'db_guard.py'
s = db.read_text(encoding='utf-8')
if 'from .order_process_state_helper import apply_order_helper' not in s:
    marker = 'from .canonical_resolver import build_canonical_cache\n'
    s = s.replace(marker, marker + '''try:\n    from .order_process_state_helper import apply_order_helper\nexcept Exception:\n    def apply_order_helper(calls_obj, state, turn):\n        return calls_obj, [], []\ntry:\n    from .kitchen_branch_helper import apply_kitchen_helper\nexcept Exception:\n    def apply_kitchen_helper(calls_obj, state, turn):\n        return calls_obj, [], []\ntry:\n    from .v8_event_logger import write_v8_event\nexcept Exception:\n    def write_v8_event(*args, **kwargs):\n        return None\n''')
needle = '''    batch_has_order_remove = any(\n        isinstance(c, dict)\n        and action_family(str(c.get("tool_name", ""))) == "remove"\n        and "order" in str(c.get("tool_name", "")).lower()\n        for c in calls\n    )\n\n    for call in calls:\n'''
replacement = '''    batch_has_order_remove = any(\n        isinstance(c, dict)\n        and action_family(str(c.get("tool_name", ""))) == "remove"\n        and "order" in str(c.get("tool_name", "")).lower()\n        for c in calls\n    )\n\n    if scenario == "order":\n        tool_call_obj, v8_synth, v8_decisions = apply_order_helper(tool_call_obj, state, turn)\n        synthetic_results.extend(v8_synth)\n        decisions.extend(v8_decisions)\n        calls = tool_call_obj if isinstance(tool_call_obj, list) else ([tool_call_obj] if tool_call_obj else [])\n    elif scenario == "kitchen":\n        tool_call_obj, v8_synth, v8_decisions = apply_kitchen_helper(tool_call_obj, state, turn)\n        synthetic_results.extend(v8_synth)\n        decisions.extend(v8_decisions)\n        calls = tool_call_obj if isinstance(tool_call_obj, list) else ([tool_call_obj] if tool_call_obj else [])\n\n    for call in calls:\n'''
if needle in s:
    s = s.replace(needle, replacement)
else:
    raise SystemExit('db_guard insertion point not found')
db.write_text(s, encoding='utf-8')

# Patch service wrapper to inspect natural replies for order final-stage process issues.
sw = WRAP / 'service_agent_wrapper.py'
s = sw.read_text(encoding='utf-8')
if 'from .order_process_state_helper import inspect_natural_reply' not in s:
    s = s.replace('''try:\n    from .human_prior_controller import observe_model_reply, observe_validated_output\nexcept Exception:\n''', '''try:\n    from .human_prior_controller import observe_model_reply, observe_validated_output\nexcept Exception:\n''')
    s = s.replace('''    def observe_validated_output(*args, **kwargs):\n        return None\n\n\nCODEX_ROOT''', '''    def observe_validated_output(*args, **kwargs):\n        return None\ntry:\n    from .order_process_state_helper import inspect_natural_reply\nexcept Exception:\n    def inspect_natural_reply(reply, state, turn):\n        return {"allow": True}\n\n\nCODEX_ROOT''')
needle = '''    if not _looks_like_tool_attempt(reply):\n        _append_log(record)\n        append_wrapper_event(episode_state, {\n'''
replacement = '''    if not _looks_like_tool_attempt(reply):\n        natural_decision = inspect_natural_reply(reply, episode_state or {}, turn)\n        if not natural_decision.get("allow", True):\n            reply = natural_decision.get("replacement") or reply\n            record["final_action"] = reply\n            record["natural_reply_rewritten"] = True\n        _append_log(record)\n        append_wrapper_event(episode_state, {\n'''
if needle in s:
    s = s.replace(needle, replacement)
else:
    raise SystemExit('service wrapper insertion point not found')
sw.write_text(s, encoding='utf-8')

# Patch prompt builder with short V8 helper instructions controlled by env.
pb = WRAP / 'prompt_builder.py'
s = pb.read_text(encoding='utf-8')
if 'V8 helper policy' not in s:
    s += '''\n\ndef v8_helper_prompt(scenario: str) -> str:\n    import os\n    bits = []\n    if scenario == "order" and os.environ.get("TRACK2_ENABLE_ORDER_HELPER", "0") == "1":\n        bits.append("V8 helper policy for order: pin restaurant before mutation/aggregate; use dish tools for dish_name and set-meal tools for set_meal_name; after add/remove use the needed aggregate tool; do not ask visual-detail questions when visual evidence is unavailable; retrieve within pinned restaurant instead.")\n    if scenario == "kitchen" and os.environ.get("TRACK2_ENABLE_KITCHEN_HELPER", "0") == "1":\n        bits.append("V8 helper policy for kitchen: follow recipe branch, get recipe ingredients once, avoid broad scans, query branch-critical quantities instead of inventing numbers, and compute nutrition only from confirmed provenance.")\n    return "\\n".join(bits)\n'''
    # Insert into enhance_prompt return path by replacing final return if simple enough.
    s = s.replace('''    return text\n''', '''    extra = v8_helper_prompt(scenario)\n    if extra:\n        text += "\\n\\n" + extra\n    return text\n''', 1)
pb.write_text(s, encoding='utf-8')

ts = time.strftime('%Y%m%d_%H%M%S')
report = REPORTS / f'V8_IMPLEMENTATION_{ts}.md'
report.write_text(f'''# V8 Implementation {ts}\n\n- Added V8 event logger: `wrappers/egobench_agent_plus/v8_event_logger.py`.\n- Added order process helper: `order_process_state_helper.py`.\n- Added kitchen branch helper: `kitchen_branch_helper.py`.\n- Added DeepSeek cross-validator plumbing: `deepseek_cross_validator.py` (online calls disabled unless explicitly enabled).\n- Added deterministic multicandidate reranker scaffold: `multicandidate_reranker.py`.\n- Wired order/kitchen helpers into `db_guard.apply_pre_execution_guard`.\n- Wired natural-reply order process check into `service_agent_wrapper.maybe_repair_agent_reply`.\n- Added short V8 helper policies in `prompt_builder` behind env switches.\n- All helper modules are switch-controlled and log to `runs/{{version}}/{{run_id}}/v8_events/{{task_id}}.jsonl`.\n- Current best was not overwritten. No final submission was made.\n''', encoding='utf-8')
with (CODEX / 'README_STATUS.md').open('a', encoding='utf-8') as f:
    f.write(f'\n## V8 Implementation {ts}\n\n- Report: `{report}`\n- Added switch-controlled order/kitchen helpers, V8 event logger, DeepSeek crosscheck scaffold, and multicandidate reranker scaffold.\n- Current best remains V6_1_3 unless validation_A/B both improve.\n- No final submission was made.\n')
print(report)
