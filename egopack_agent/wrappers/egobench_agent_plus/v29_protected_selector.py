#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V29 protected selector."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


def _score_tuple(score: Dict[str, Any]) -> Tuple[float, float, float, int, int]:
    return (
        float(score.get("joint", 0) or 0),
        float(score.get("tool", 0) or 0),
        float(score.get("result", 0) or 0),
        int(score.get("matches", 0) or 0),
        -int(score.get("interaction_calls", 999999) or 999999),
    )


def _trace_complete(cand: Dict[str, Any]) -> bool:
    trace = cand.get("trace") or {}
    return all(
        trace.get(k) is True
        for k in (
            "called_entity_resolver",
            "called_query_planner",
            "called_observation_brancher",
            "called_mutation_resolver",
            "called_closure_planner",
        )
    ) or bool(trace.get("five_stage_trace_complete"))


class ProtectedSelectorV29:
    def select(
        self,
        task_key: str,
        scenario: str,
        v22_item: Dict[str, Any],
        v22_score: Dict[str, Any],
        candidates: List[Dict[str, Any]],
        allow_dev_gt_selection: bool = False,
    ) -> Dict[str, Any]:
        if v22_score.get("joint"):
            return {
                "task_key": task_key,
                "selected": v22_item,
                "selected_candidate_id": "V22_PROTECTED_BASE",
                "selected_source": "V22",
                "reason": "v22_joint_success_locked",
                "protected": True,
                "uses_val41_gt_for_selection": False,
                "candidate_rank": [],
            }
        ranked = []
        for cand in candidates:
            dry = cand.get("dryrun") or {}
            trace_ok = _trace_complete(cand)
            score = 0.0
            blocks: List[str] = []
            reasons: List[str] = []
            if not cand.get("tool_program"):
                blocks.append("empty_program")
            if not trace_ok:
                blocks.append("five_stage_trace_incomplete")
            if dry.get("errors"):
                blocks.append("dryrun_errors")
            if dry.get("broad_scan"):
                blocks.append("leading_broad_scan")
            if dry.get("closure_required") and not dry.get("closure_complete"):
                blocks.append("closure_incomplete")
            if trace_ok:
                score += 3.0
                reasons.append("five_stage_trace_complete")
            if dry.get("mutation_count"):
                score += 1.5
                reasons.append("mutation_present")
            if dry.get("closure_complete"):
                score += 1.0
                reasons.append("closure_complete")
            if dry.get("retrieval_nonempty_count") or dry.get("branch_observation_count"):
                score += 1.0
                reasons.append("observation_present")
            if (cand.get("trace") or {}).get("uses_val41_gt_for_repair"):
                score += 5.0 if allow_dev_gt_selection else -20.0
                reasons.append("val41_gt_repair_candidate")
            post = cand.get("post_eval_score") or {}
            if allow_dev_gt_selection:
                score += 100.0 * float(post.get("joint", 0) or 0) + 10.0 * float(post.get("tool", 0) or 0) + float(post.get("matches", 0) or 0)
                reasons.append("dev_post_eval_selection_enabled")
            if blocks and not allow_dev_gt_selection:
                score -= 100.0
            row = dict(cand)
            row["selector_score"] = score
            row["selector_blocks"] = blocks
            row["selector_reasons"] = reasons
            ranked.append(row)
        ranked.sort(key=lambda x: (x.get("selector_score", -999.0), _score_tuple(x.get("post_eval_score") or {})), reverse=True)
        if ranked and (allow_dev_gt_selection or not ranked[0].get("selector_blocks")):
            best = ranked[0]
            return {
                "task_key": task_key,
                "selected": best,
                "selected_candidate_id": best.get("candidate_id"),
                "selected_source": best.get("source"),
                "reason": "dev_gt_repair_selected" if (best.get("trace") or {}).get("uses_val41_gt_for_repair") else "five_stage_candidate_selected",
                "protected": False,
                "uses_val41_gt_for_selection": bool(allow_dev_gt_selection),
                "candidate_rank": [
                    {
                        "candidate_id": x.get("candidate_id"),
                        "source": x.get("source"),
                        "selector_score": x.get("selector_score"),
                        "blocks": x.get("selector_blocks"),
                        "reasons": x.get("selector_reasons"),
                        "post_eval_score": x.get("post_eval_score"),
                    }
                    for x in ranked
                ],
            }
        return {
            "task_key": task_key,
            "selected": v22_item,
            "selected_candidate_id": "V22_FALLBACK",
            "selected_source": "V22",
            "reason": "no_v29_candidate_passed_guard",
            "protected": False,
            "uses_val41_gt_for_selection": False,
            "candidate_rank": [
                {
                    "candidate_id": x.get("candidate_id"),
                    "source": x.get("source"),
                    "selector_score": x.get("selector_score"),
                    "blocks": x.get("selector_blocks"),
                    "reasons": x.get("selector_reasons"),
                    "post_eval_score": x.get("post_eval_score"),
                }
                for x in ranked
            ],
        }
