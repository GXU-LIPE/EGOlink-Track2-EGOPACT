#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Protected delta selector for V28."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


MUTATION_TOOLS = {
    "add_to_cart",
    "remove_from_cart",
    "add_dish_to_order",
    "add_set_meal_to_order",
    "remove_dish_from_order",
    "remove_set_meal_from_order",
    "add_to_shopping_list",
    "remove_from_shopping_list",
    "add_recipe_to_menu",
    "remove_recipe_from_menu",
}


def _names(program: List[Dict[str, Any]]) -> List[str]:
    return [str(c.get("tool_name", "")) for c in program or []]


def _score_candidate(candidate: Dict[str, Any], dryrun: Dict[str, Any], guard: Dict[str, Any]) -> Tuple[float, List[str], List[str]]:
    program = candidate.get("tool_program") or []
    names = _names(program)
    score = 0.0
    why: List[str] = []
    blocks: List[str] = []
    if not program:
        blocks.append("empty_program")
    if not candidate.get("takeover_certified"):
        blocks.append("not_delta_certified")
    if dryrun.get("errors"):
        blocks.append("dryrun_errors")
    if dryrun.get("broad_scan"):
        blocks.append("leading_broad_scan")
    if dryrun.get("closure_required") and not dryrun.get("closure_complete"):
        blocks.append("closure_incomplete")
    if guard.get("allow") is False:
        blocks.append("evidence_guard_veto")
    if any(n in MUTATION_TOOLS for n in names):
        score += 2.0
        why.append("mutation_present")
    if dryrun.get("retrieval_nonempty_count") or dryrun.get("branch_observation_count"):
        score += min(2.5, float(dryrun.get("retrieval_nonempty_count", 0)) * 0.3 + float(dryrun.get("branch_observation_count", 0)) * 0.6)
        why.append("tool_observation_present")
    if dryrun.get("closure_complete"):
        score += 1.5
        why.append("closure_complete")
    if names and names[0].startswith(("get_", "find_", "filter_", "list_")):
        score += 0.5
        why.append("query_first")
    if guard.get("action") in {"tiebreak", "query_hint"}:
        score += 0.3
        why.append("evidence_safe_auxiliary")
    if "evidence_disagrees_with_mutation_target" in (guard.get("risk_flags") or []):
        score -= 2.0
        why.append("evidence_target_disagreement")
    if len(program) > 30:
        score -= 1.0
        why.append("tool_count_high")
    if blocks:
        score -= 100.0
    return score, why, blocks


class ProtectedDeltaSelectorV28:
    """Delta-only selector: V22 joint successes are locked."""

    def select(
        self,
        task_key: str,
        scenario: str,
        v22_item: Dict[str, Any],
        v22_score: Dict[str, Any],
        candidates: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if v22_score.get("joint"):
            return {
                "task_key": task_key,
                "selected": v22_item,
                "selected_candidate_id": "V22_PROTECTED_BASE",
                "selected_source": "V22",
                "reason": "v22_joint_success_locked",
                "protected": True,
                "candidate_rank": [],
            }
        ranked = []
        for cand in candidates:
            score, why, blocks = _score_candidate(cand, cand.get("dryrun") or {}, cand.get("evidence_guard") or {})
            row = dict(cand)
            row["selector_score"] = score
            row["selector_reasons"] = why
            row["selector_blocks"] = blocks
            ranked.append(row)
        ranked.sort(key=lambda x: (x.get("selector_score", -999), -len(x.get("tool_program") or [])), reverse=True)
        if ranked and not ranked[0].get("selector_blocks"):
            best = ranked[0]
            return {
                "task_key": task_key,
                "selected": best,
                "selected_candidate_id": best.get("candidate_id"),
                "selected_source": best.get("source"),
                "reason": "delta_candidate_guard_passed",
                "protected": False,
                "candidate_rank": [
                    {
                        "candidate_id": x.get("candidate_id"),
                        "source": x.get("source"),
                        "selector_score": x.get("selector_score"),
                        "blocks": x.get("selector_blocks"),
                        "reasons": x.get("selector_reasons"),
                    }
                    for x in ranked
                ],
            }
        return {
            "task_key": task_key,
            "selected": v22_item,
            "selected_candidate_id": "V22_FALLBACK",
            "selected_source": "V22",
            "reason": "no_delta_candidate_passed_guard",
            "protected": False,
            "candidate_rank": [
                {
                    "candidate_id": x.get("candidate_id"),
                    "source": x.get("source"),
                    "selector_score": x.get("selector_score"),
                    "blocks": x.get("selector_blocks"),
                    "reasons": x.get("selector_reasons"),
                }
                for x in ranked
            ],
        }
