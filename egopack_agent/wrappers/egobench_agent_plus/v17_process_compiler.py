# -*- coding: utf-8 -*-
"""V17 executable compiler cards and lightweight tool-call repair.

The compiler is deliberately narrow. It uses GT100-distilled indexes from
non-final, non-val41 data to make generic process repairs, but it never reads
val41 labels or final hidden metadata.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


CODEX_ROOT = Path(os.environ.get("CODEX_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/codex"))
KNOWN_RESTAURANTS = (
    "Annie Italian Restaurant",
    "Mediterranean Greek Restaurant",
    "Afrikana Restaurant",
    "Butcher Restaurant",
    "Meraki Restaurant",
    "Pauhana Restaurant",
    "Sunny Restaurant",
)


def _load(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return default


def _canonical_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s&]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _canon_restaurant(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return raw
    raw_c = _canonical_text(raw)
    for name in KNOWN_RESTAURANTS:
        if _canonical_text(name) == raw_c or _canonical_text(name) in raw_c:
            return name
    return raw


def _as_calls(normalized: str) -> Tuple[List[Dict[str, Any]], bool]:
    try:
        obj = json.loads(normalized)
    except Exception:
        return [], False
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)], True
    if isinstance(obj, dict):
        return [obj], False
    return [], True


def _dump_calls(calls: List[Dict[str, Any]], was_list: bool) -> str:
    return json.dumps(calls if was_list else (calls[0] if calls else []), ensure_ascii=False)


def _is_compute(name: str) -> bool:
    return name.startswith(("compute_total_payment", "compute_total_tax", "compute_total_price", "compute_total_nutrition", "compute_total_nutritions"))


def _is_mutation(name: str) -> bool:
    if _is_compute(name):
        return False
    return (
        name.startswith(("add", "remove", "delete", "update", "modify"))
        or "_to_cart" in name
        or "_from_cart" in name
        or "_to_order" in name
        or "_from_order" in name
        or "_to_shopping_list" in name
        or "_to_menu" in name
    )


def _needs_order_restaurant(name: str) -> bool:
    return any(piece in name for piece in ("dish", "set_meal", "order", "tax", "payment", "nutrition", "price"))


def _retail_broad_scan(params: Dict[str, Any]) -> bool:
    if not isinstance(params, dict):
        return False
    low = params.get("min_price", params.get("price_min", params.get("lower_price")))
    high = params.get("max_price", params.get("price_max", params.get("upper_price")))
    try:
        low_f = float(low) if low not in (None, "") else None
        high_f = float(high) if high not in (None, "") else None
    except Exception:
        return False
    return low_f is not None and high_f is not None and low_f <= 0 and high_f >= 10000


def repair_tool_output(normalized: str, scenario: str, state: Dict[str, Any], turn: int = 0) -> Tuple[str, Dict[str, Any]]:
    """Apply generic executable-process repairs to validated tool JSON.

    Repairs are intentionally conservative:
    - canonicalize/autofill pinned order restaurant where existing guards would
      otherwise need to do it later;
    - drop obvious retail broad numeric scans when they appear in the same
      batch as narrower candidate retrievals;
    - record repair telemetry for smoke analysis.
    """
    if os.environ.get("TRACK2_ENABLE_V17_COMPILER") != "1" and not str(os.environ.get("TRACK2_RUN_VERSION", "")).startswith("V17_"):
        return normalized, {"enabled": False}
    if os.environ.get("TRACK2_V17_EXEC_REPAIR", "1") == "0":
        return normalized, {"enabled": True, "changed": False, "exec_repair": False}
    calls, was_list = _as_calls(normalized)
    if not calls:
        return normalized, {"enabled": True, "changed": False, "reason": "parse_empty"}
    report: Dict[str, Any] = {
        "enabled": True,
        "changed": False,
        "turn": turn,
        "scenario": scenario,
        "repairs": [],
    }
    pins = (state or {}).get("pins") or {}
    pinned_restaurant = _canon_restaurant(pins.get("restaurant_name"))
    repaired: List[Dict[str, Any]] = []
    narrow_retail_batch = False
    if scenario == "retail":
        for call in calls:
            params = call.get("parameters", {}) if isinstance(call.get("parameters"), dict) else {}
            if any(params.get(k) for k in ("product_name", "category", "brand", "country", "origin", "taste", "profile")) and not _retail_broad_scan(params):
                narrow_retail_batch = True
                break
    for call in calls:
        call2 = dict(call)
        name = str(call2.get("tool_name") or "")
        lname = name.lower()
        params = dict(call2.get("parameters") or {}) if isinstance(call2.get("parameters"), dict) else {}
        if scenario == "order":
            if "restaurant_name" in params and params.get("restaurant_name"):
                canon = _canon_restaurant(params.get("restaurant_name"))
                if canon != params.get("restaurant_name"):
                    report["changed"] = True
                    report["repairs"].append({"tool_name": name, "repair": "canonical_restaurant", "from": params.get("restaurant_name"), "to": canon})
                    params["restaurant_name"] = canon
            if pinned_restaurant and _needs_order_restaurant(lname) and not params.get("restaurant_name"):
                params["restaurant_name"] = pinned_restaurant
                report["changed"] = True
                report["repairs"].append({"tool_name": name, "repair": "autofill_pinned_restaurant", "value": pinned_restaurant})
            if lname in {"compute_total_payment", "compute_total_tax", "compute_total_price"}:
                dishes = params.get("dishes")
                if isinstance(dishes, list):
                    changed_items = 0
                    for item in dishes:
                        if isinstance(item, dict) and "dish_name" in item and "product_name" not in item:
                            item["product_name"] = item.pop("dish_name")
                            changed_items += 1
                    if changed_items:
                        report["changed"] = True
                        report["repairs"].append({"tool_name": name, "repair": "order_aggregate_dish_name_to_product_name", "count": changed_items})
        if scenario == "retail" and narrow_retail_batch and _retail_broad_scan(params):
            report["changed"] = True
            report["repairs"].append({"tool_name": name, "repair": "drop_retail_broad_scan", "parameters": params})
            continue
        call2["parameters"] = params
        repaired.append(call2)
    if not repaired:
        report["changed"] = False
        report["all_calls_dropped"] = True
        return normalized, report
    if report["changed"]:
        try:
            from .db_guard import append_wrapper_event
            append_wrapper_event(state, {"event": "v17_compiler_repair", "turn": turn, "report": report})
        except Exception:
            pass
        return _dump_calls(repaired, was_list), report
    return normalized, report


def _intent(goal: str, scenario: str) -> str:
    text = (goal or "").lower()
    if any(x in text for x in ["if ", "otherwise", "else", "whether", "exceeds", "less than", "greater than"]):
        return "branch_then_mutation"
    if any(x in text for x in ["highest", "lowest", "cheapest", "most", "least"]):
        return "ranking_filtering"
    if any(x in text for x in ["add", "remove", "cart", "order", "menu", "shopping list"]):
        return "cart_order_mutation"
    if any(x in text for x in ["total", "tax", "payment", "nutrition", "summary"]):
        return "aggregate_required"
    if any(x in text for x in ["point", "visible", "image", "video", "shelf"]):
        return "visual_query"
    return "query_only"


def _select(index: Dict[str, Any], scenario: str, intent: str, limit: int = 4) -> List[Dict[str, Any]]:
    data = index.get("index", {})
    out = []
    for key in [f"{scenario}::{intent}", f"{scenario}::branch_then_mutation", f"{scenario}::cart_order_mutation"]:
        out.extend(data.get(key, []))
    return out[:limit]


def build_v17_compiler_prompt(scenario: str) -> str:
    if os.environ.get("TRACK2_ENABLE_V17_COMPILER") != "1":
        return ""
    bank = Path(os.environ.get("TRACK2_V17_DISTILL_DIR") or (CODEX_ROOT / "gt_distill_v17"))
    goal = os.environ.get("TRACK2_CURRENT_USER_GOAL", "")
    intent = _intent(goal, scenario)
    skeleton = _load(bank / "tool_skeleton_index.json", {})
    slots = _load(bank / "slot_resolver_index.json", {})
    branch = _load(bank / "branch_compiler_index.json", {})
    closure = _load(bank / "closure_repair_index.json", {})
    anti = _load(bank / "anti_broad_scan_index.json", {})
    skels = _select(skeleton, scenario, intent, 3)
    slot_cards = _select(slots, scenario, intent, 3)
    branch_cards = _select(branch, scenario, intent, 2)
    closure_cards = _select(closure, scenario, intent, 2)
    lines = [
        "[V17 GT100 Executable Compiler]",
        f"- intent_type: {intent}",
        "- Source: 670 valid GPT-5.5 distilled rules from non-final, non-val41 GT100 pool. No final hidden metadata. No val41 labels.",
        "- This is not natural-language memory. Before each tool call, enforce: slot resolution -> canonicalization -> branch check if needed -> mutation once -> aggregate/summary closure if requested.",
        "- If GPT candidate lacks required process closure, repair the next action rather than asking the user or broad-scanning.",
    ]
    if skels:
        lines.append("- Skeleton candidates:")
        for item in skels:
            stages = item.get("stages") or []
            desc = []
            for s in stages[:8]:
                if isinstance(s, dict):
                    desc.append(f"{s.get('stage')}[{','.join(s.get('allowed_tools') or [])}]")
                else:
                    desc.append(str(s))
            lines.append("  * " + " -> ".join(desc))
    if slot_cards:
        lines.append("- Slot resolver cards:")
        for item in slot_cards:
            req = item.get("required_slots") or {}
            keep = {k: v for k, v in req.items() if v not in ("", None, [], {})}
            if keep:
                lines.append("  * required_slots: " + json.dumps(keep, ensure_ascii=False)[:500])
    if branch_cards:
        lines.append("- Branch compiler cards:")
        for item in branch_cards:
            lines.append("  * " + json.dumps(item.get("rules") or [], ensure_ascii=False)[:600])
    if closure_cards:
        lines.append("- Closure repair cards:")
        for item in closure_cards:
            lines.append("  * " + json.dumps(item.get("rules") or [], ensure_ascii=False)[:600])
    anti_cards = (anti.get("index") or {}).get(scenario, [])[:8]
    if anti_cards:
        lines.append("- Anti-patterns to avoid:")
        for item in anti_cards:
            lines.append(f"  * {item.get('anti_pattern')}")
    if scenario == "order":
        lines += [
            "- Order compiler hard rules:",
            "  * Maintain active restaurant ledger. After 'from now on/use that restaurant', all later order tools use that restaurant_name.",
            "  * Separate dish_name and set_meal_name. If set meal membership matters, call get_set_meal_details before mutation.",
            "  * If mutation occurred and total/payment/tax/nutrition is requested, close with get_user_order_summary and one matching compute_total_* call.",
        ]
    if scenario == "retail":
        lines += [
            "- Retail compiler hard rules:",
            "  * Visual phrase -> candidate product set -> numeric attribute only for narrowed candidates.",
            "  * Broad 0-100000 price scan is disallowed for visual grounding.",
        ]
    return "\n".join(lines)
