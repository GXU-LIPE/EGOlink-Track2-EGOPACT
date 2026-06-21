#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Protected selector for V30 prior-agent candidates."""

from __future__ import annotations

from typing import Any, Dict, List


class ProtectedPriorSelectorV30:
    def select(self, task_key: str, v22_item: Dict[str, Any], v22_score: Dict[str, Any], candidates: List[Dict[str, Any]], mode: str) -> Dict[str, Any]:
        if v22_score.get("joint"):
            return {
                "task_key": task_key,
                "selected": v22_item,
                "selected_candidate_id": "V22_PROTECTED_BASE",
                "selected_source": "V22",
                "reason": "v22_joint_success_locked",
                "protected": True,
                "uses_post_eval_for_selection": False,
            }
        ranked = []
        for cand in candidates:
            dry = cand.get("dryrun") or {}
            post = cand.get("post_eval_score") or {}
            score = float(cand.get("retrieval_score", 0.0))
            reasons = ["retrieval_score"]
            blocks: List[str] = []
            if not cand.get("tool_program"):
                blocks.append("empty_program")
            if dry.get("errors"):
                blocks.append("dryrun_errors")
            if dry.get("broad_scan"):
                blocks.append("leading_broad_scan")
            if dry.get("closure_required") and not dry.get("closure_complete"):
                blocks.append("closure_incomplete")
            if dry.get("mutation_count"):
                score += 0.6
                reasons.append("mutation_present")
            if dry.get("closure_complete"):
                score += 0.5
                reasons.append("closure_complete")
            if cand.get("slot_source") == "retrieved_dev_experience_case":
                score += 0.8
                reasons.append("dev_experience_slots_available")
            if mode == "dev_calibrated":
                score += 100.0 * float(post.get("joint", 0) or 0) + 10.0 * float(post.get("tool", 0) or 0) + float(post.get("matches", 0) or 0)
                reasons.append("dev_post_eval_prior_calibration")
            if blocks and mode != "dev_calibrated":
                score -= 100.0
            row = dict(cand)
            row["selector_score"] = score
            row["selector_reasons"] = reasons
            row["selector_blocks"] = blocks
            ranked.append(row)
        ranked.sort(key=lambda x: (x.get("selector_score", -999.0), x.get("retrieval_score", 0.0)), reverse=True)
        if ranked and (mode == "dev_calibrated" or not ranked[0].get("selector_blocks")):
            best = ranked[0]
            return {
                "task_key": task_key,
                "selected": best,
                "selected_candidate_id": best.get("candidate_id"),
                "selected_source": best.get("source"),
                "selected_prior_id": best.get("prior_id"),
                "selected_program_family": best.get("program_family"),
                "reason": "dev_calibrated_prior_selected" if mode == "dev_calibrated" else "prior_candidate_guard_passed",
                "protected": False,
                "uses_post_eval_for_selection": mode == "dev_calibrated",
                "candidate_rank": [
                    {
                        "candidate_id": x.get("candidate_id"),
                        "prior_id": x.get("prior_id"),
                        "program_family": x.get("program_family"),
                        "retrieval_score": x.get("retrieval_score"),
                        "selector_score": x.get("selector_score"),
                        "blocks": x.get("selector_blocks"),
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
            "reason": "no_prior_candidate_passed_guard",
            "protected": False,
            "uses_post_eval_for_selection": False,
            "candidate_rank": [
                {
                    "candidate_id": x.get("candidate_id"),
                    "prior_id": x.get("prior_id"),
                    "program_family": x.get("program_family"),
                    "retrieval_score": x.get("retrieval_score"),
                    "selector_score": x.get("selector_score"),
                    "blocks": x.get("selector_blocks"),
                    "post_eval_score": x.get("post_eval_score"),
                }
                for x in ranked
            ],
        }
