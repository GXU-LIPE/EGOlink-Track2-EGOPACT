#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Guarded selector for V31 LOO diagnostics."""

from __future__ import annotations

from typing import Any, Dict, List


def _score_candidate(cand: Dict[str, Any], v22_score: Dict[str, Any]) -> float:
    dry = cand.get("dryrun") or {}
    score = float(cand.get("retrieval_score") or 0.0)
    score += float(cand.get("slot_confidence") or 0.0) * 2.0
    if dry.get("ok"):
        score += 2.5
    if dry.get("mutation_count"):
        score += 1.5
    if dry.get("aggregate_count"):
        score += 1.0
    if dry.get("closure_complete"):
        score += 1.2
    if dry.get("retrieval_nonempty_count") or dry.get("branch_observation_count"):
        score += 0.8
    if dry.get("broad_scan"):
        score -= 5.0
    if dry.get("errors"):
        score -= 10.0
    risks = cand.get("risk_flags") or []
    score -= 0.4 * len(risks)
    if cand.get("source") == "V22":
        score += 100.0 if v22_score.get("joint") else 0.0
    return score


class V31LOOSelector:
    def select(self, task_key: str, v22_item: Dict[str, Any], v22_score: Dict[str, Any], candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        ranked: List[Dict[str, Any]] = []
        for cand in candidates:
            c = dict(cand)
            dry = c.get("dryrun") or {}
            blocks: List[str] = []
            if not c.get("tool_program"):
                blocks.append("empty_program")
            if dry.get("errors"):
                blocks.append("dryrun_errors")
            if dry.get("broad_scan"):
                blocks.append("broad_scan")
            if "missing_slot_values" in (c.get("risk_flags") or []):
                blocks.append("missing_slot_values")
            c["selector_score"] = _score_candidate(c, v22_score)
            c["selector_blocks"] = blocks
            ranked.append(c)
        ranked.sort(key=lambda x: (x.get("selector_score", -999), x.get("retrieval_score", 0.0)), reverse=True)

        if v22_score.get("joint"):
            v22_cand = next((c for c in ranked if c.get("source") == "V22"), None)
            return {
                "task_key": task_key,
                "selected": v22_cand or {"source": "V22", "tool_program": []},
                "selected_candidate_id": "V22_PROTECTED_BASE",
                "selected_source": "V22",
                "reason": "v22_joint_success_locked",
                "protected": True,
                "uses_current_heldout_gt_for_selection": False,
                "candidate_rank": self._rank_view(ranked),
            }

        for cand in ranked:
            if cand.get("source") == "V22":
                continue
            if cand.get("selector_blocks"):
                continue
            if cand.get("source") in {"V31_LOO_SLOT_FILLER", "V31_CLOSURE_REPAIR"}:
                return {
                    "task_key": task_key,
                    "selected": cand,
                    "selected_candidate_id": cand.get("candidate_id"),
                    "selected_source": cand.get("source"),
                    "selected_prior_id": cand.get("prior_id"),
                    "selected_program_family": cand.get("program_family"),
                    "reason": "v31_guard_passed",
                    "protected": False,
                    "uses_current_heldout_gt_for_selection": False,
                    "candidate_rank": self._rank_view(ranked),
                }
        v22_cand = next((c for c in ranked if c.get("source") == "V22"), None)
        return {
            "task_key": task_key,
            "selected": v22_cand or {"source": "V22", "tool_program": []},
            "selected_candidate_id": "V22_FALLBACK",
            "selected_source": "V22",
            "reason": "no_v31_candidate_passed_guard",
            "protected": False,
            "uses_current_heldout_gt_for_selection": False,
            "candidate_rank": self._rank_view(ranked),
        }

    @staticmethod
    def _rank_view(ranked: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            {
                "candidate_id": c.get("candidate_id"),
                "source": c.get("source"),
                "prior_id": c.get("prior_id"),
                "program_family": c.get("program_family"),
                "retrieval_score": c.get("retrieval_score"),
                "slot_confidence": c.get("slot_confidence"),
                "selector_score": c.get("selector_score"),
                "blocks": c.get("selector_blocks"),
                "risk_flags": c.get("risk_flags"),
                "dryrun": c.get("dryrun"),
            }
            for c in ranked[:10]
        ]
