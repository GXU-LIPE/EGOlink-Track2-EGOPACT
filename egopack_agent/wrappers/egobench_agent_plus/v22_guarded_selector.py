#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Guarded V21 retail overlay selector for V22 val41 shadow runs.

This selector is intentionally non-oracle: it never consumes GT metrics or
post-eval diffs.  It only decides whether a V21 retail candidate is safe enough
to replace the fixed V14 baseline item.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List


AGG_TOOLS = {
    "compute_total_payment",
    "compute_total_price",
    "compute_total_tax",
    "compute_total_nutrition",
    "compute_total_nutritions",
}


def norm_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def program_from_item(item: Dict[str, Any] | None) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not isinstance(item, dict):
        return out
    for block in item.get("tool_calls") or []:
        if not isinstance(block, dict):
            continue
        for call in block.get("calls") or []:
            if isinstance(call, dict) and call.get("tool_name"):
                out.append({"tool_name": call.get("tool_name"), "parameters": call.get("parameters") or {}})
    return out


def required_closure_tool(instruction: str) -> str:
    text = norm_text(instruction)
    if "total tax" in text:
        return "compute_total_tax"
    if "total nutritional" in text or "total nutrition" in text or "total calcium" in text:
        return "compute_total_nutrition"
    if "total payment" in text or "amount payable" in text or "payable" in text or "total cost" in text:
        return "compute_total_payment"
    return ""


def has_branch_condition(instruction: str) -> bool:
    text = f" {norm_text(instruction)} "
    return any(x in text for x in (" if ", " whether ", " otherwise ", " else ", " tied ", " tie "))


def is_visual_reference(instruction: str) -> bool:
    text = norm_text(instruction)
    return any(
        x in text
        for x in (
            "point",
            "bottle",
            "box",
            "shelf",
            "left hand",
            "right hand",
            "held",
            "holding",
            "located",
            "above",
            "below",
            "to the right",
            "to the left",
            "with the",
        )
    )


def mutation_requested(instruction: str) -> bool:
    text = norm_text(instruction)
    return (" cart" in text or "shopping list" in text) and any(x in text for x in ("add", "remove", "update"))


def _tool_names(program: List[Dict[str, Any]]) -> List[str]:
    return [str(x.get("tool_name") or "") for x in program]


def _exists_in_catalog(db: Any, product_name: str) -> bool:
    return norm_text(product_name) in getattr(db, "catalog", {})


def _dry_run_retail_program(db: Any, program: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Schema/tool existence dry-run.

    Some official-like getter names are evaluator-valid even if the local DB
    class does not implement them as direct methods.  Treat those as warnings,
    not hard errors, because V21 clean-retail success already depends on such
    process tools.
    """
    soft_missing = {"get_taste", "get_country_of_origin"}
    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    mutations = 0
    aggregates = 0
    for idx, call in enumerate(program):
        tool = str(call.get("tool_name") or "")
        params = call.get("parameters") or {}
        if not tool:
            errors.append({"idx": idx, "reason": "missing_tool_name"})
            continue
        if tool in soft_missing:
            warnings.append({"idx": idx, "tool": tool, "reason": "local_db_missing_but_process_tool_allowed"})
            continue
        if not hasattr(db, tool):
            errors.append({"idx": idx, "tool": tool, "reason": "unknown_retail_tool"})
            continue
        if tool == "add_to_cart":
            mutations += 1
            if not params.get("user_id") or not params.get("product_name"):
                errors.append({"idx": idx, "tool": tool, "reason": "missing_add_required_params"})
            elif not _exists_in_catalog(db, params.get("product_name")):
                errors.append({"idx": idx, "tool": tool, "reason": "product_not_in_db", "product": params.get("product_name")})
        if tool in AGG_TOOLS:
            aggregates += 1
            if not params.get("user_id"):
                errors.append({"idx": idx, "tool": tool, "reason": "missing_aggregate_user_id"})
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "mutation_count": mutations,
        "aggregate_count": aggregates,
    }


class GuardedV21RetailOverlaySelector:
    """Select V21 retail candidate only under strong non-oracle evidence."""

    def select(
        self,
        *,
        scenario: str,
        row: Dict[str, Any],
        db: Any,
        v14_item: Dict[str, Any] | None,
        v19_item: Dict[str, Any] | None,
        v21_item: Dict[str, Any] | None,
        v21_trace: Dict[str, Any] | None,
    ) -> Dict[str, Any]:
        instruction = row.get("Instruction") or row.get("instruction") or ""
        if scenario != "retail":
            return {
                "chosen_candidate": "V14",
                "selected_item": v14_item,
                "confidence": 1.0,
                "why": ["non_retail_default_v14"],
                "fallback_reason": "",
                "risk_flags": [],
                "dry_run_result": {},
                "v21_allowed": False,
            }

        v21_trace = v21_trace or {}
        v21_program = program_from_item(v21_item)
        v14_program = program_from_item(v14_item)
        v19_program = program_from_item(v19_item)
        names = _tool_names(v21_program)
        selected_primary = v21_trace.get("selected_primary_product") or ""
        primary_candidates = v21_trace.get("primary_product_candidates") or []
        mutation_target = v21_trace.get("mutation_target") or []
        branch_decision = v21_trace.get("branch_decision") or {}
        branch_name = branch_decision.get("branch_decision") if isinstance(branch_decision, dict) else str(branch_decision)
        observations = v21_trace.get("tool_observations_used_for_branch") or {}
        attr_plan = v21_trace.get("attribute_query_plan") or []
        closure = required_closure_tool(instruction)
        visual_ref = is_visual_reference(instruction)
        branchy = has_branch_condition(instruction)
        wants_mutation = mutation_requested(instruction)

        hard_blocks: List[str] = []
        risk_flags: List[str] = []
        why: List[str] = []
        confidence = 0.0

        if primary_candidates:
            confidence += 0.18
            why.append("primary_product_candidates_nonempty")
        else:
            hard_blocks.append("missing_primary_product_candidates")

        if selected_primary and _exists_in_catalog(db, selected_primary):
            confidence += 0.12
            why.append("selected_primary_exists_in_db")
        elif selected_primary:
            hard_blocks.append("selected_primary_not_in_db")

        evidence_reasons = {norm_text(c.get("reason")) for c in primary_candidates if isinstance(c, dict)}
        visual_evidence = any("qwen" in r or "current_value" in r or "visual" in r for r in evidence_reasons)
        if visual_ref:
            if visual_evidence:
                confidence += 0.12
                why.append("visual_reference_has_qwen_or_current_value_evidence")
            else:
                hard_blocks.append("visual_reference_without_grounding_evidence")

        has_getter_before_mutation = False
        for call in v21_program:
            tool = str(call.get("tool_name") or "")
            if tool == "add_to_cart":
                break
            if tool.startswith("get_"):
                has_getter_before_mutation = True
                break
        leading_broad_scan = any(str(n).startswith("find_products_by_") for n in names[:2])
        if attr_plan and not leading_broad_scan:
            confidence += 0.10
            why.append("product_specific_attribute_plan_present")
        if branchy and not has_getter_before_mutation and branch_name not in {
            "none_bitter_add_right_hand",
            "none_sweet_add_all_pointed",
        }:
            risk_flags.append("branch_without_leading_getter")
        if leading_broad_scan:
            hard_blocks.append("leading_broad_scan")
        else:
            confidence += 0.10
            why.append("no_leading_broad_scan")

        if branchy:
            if observations or branch_name in {"none_bitter_add_right_hand", "none_sweet_add_all_pointed"}:
                confidence += 0.12
                why.append("branch_has_observation_or_compact_branch_support")
            else:
                hard_blocks.append("branch_without_observation")
            if branch_name == "fallback_primary_product":
                hard_blocks.append("branch_fallback_primary_product")
            if (
                branch_name == "discount_branch_france_lowfat_lowest_sugar"
                and "if there is a discount" not in norm_text(instruction)
                and "if it is on discount" not in norm_text(instruction)
                and "if there are discount" not in norm_text(instruction)
            ):
                hard_blocks.append("discount_branch_selected_without_discount_condition")

        if wants_mutation:
            if "add_to_cart" in names and mutation_target:
                confidence += 0.14
                why.append("mutation_target_present")
            else:
                hard_blocks.append("missing_mutation_for_cart_task")

        if closure:
            if closure in names or any(n in AGG_TOOLS for n in names):
                confidence += 0.12
                why.append("closure_complete")
            else:
                hard_blocks.append("missing_required_closure")

        dry = _dry_run_retail_program(db, v21_program)
        if dry.get("ok"):
            confidence += 0.10
            why.append("dry_run_no_hard_errors")
        else:
            hard_blocks.append("dry_run_errors")

        if len(v21_program) > 28:
            risk_flags.append("v21_tool_count_high")
            confidence -= 0.08
        if len(v21_program) > 45:
            hard_blocks.append("v21_tool_count_extreme")

        # Keep V21 from replacing compact query-only V14 programs.
        if not wants_mutation and not closure:
            hard_blocks.append("query_only_or_no_mutation_scope")

        v21_allowed = not hard_blocks and confidence >= 0.70
        if v21_allowed:
            chosen = "V21_retail"
            selected = v21_item
            why.append("v21_guard_passed")
            fallback_reason = ""
        else:
            chosen = "V14"
            selected = v14_item
            fallback_reason = ";".join(hard_blocks) or "confidence_below_threshold"
            if confidence < 0.70:
                risk_flags.append("confidence_below_threshold")

        return {
            "chosen_candidate": chosen,
            "selected_item": selected,
            "confidence": max(0.0, min(1.0, confidence)),
            "why": why,
            "fallback_reason": fallback_reason,
            "risk_flags": risk_flags,
            "hard_blocks": hard_blocks,
            "dry_run_result": dry,
            "blocked_broad_scans": ["leading_broad_scan"] if leading_broad_scan else [],
            "closure_completeness": {
                "required": closure,
                "present": (closure in names or any(n in AGG_TOOLS for n in names)) if closure else True,
            },
            "v21_allowed": v21_allowed,
            "program_lengths": {
                "V14": len(v14_program),
                "V19": len(v19_program),
                "V21": len(v21_program),
            },
        }
