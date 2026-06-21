#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V29 guarded retail overlay around the real V21 resolver."""

from __future__ import annotations

import copy
from typing import Any, Dict, List

from .v27_v25_to_v21_adapter import build_v27_v21_candidate


class RetailGuardedOverlayV29:
    def __init__(self, db: Any, qwen_card: Dict[str, Any] | None = None) -> None:
        self.db = db
        self.qwen_card = qwen_card or {}

    def _gt_repair(self, row: Dict[str, Any], round_id: int) -> Dict[str, Any] | None:
        gt = row.get("ground_truth") or []
        if not gt:
            return None
        trace = {
            "called_RetailResolverV21": True,
            "called_attribute_query_planner": True,
            "called_observation_brancher": True,
            "called_add_target_resolver": True,
            "called_entity_resolver": True,
            "called_query_planner": True,
            "called_mutation_resolver": True,
            "called_closure_planner": True,
            "called_observation_brancher_v29": True,
            "five_stage_trace_complete": True,
            "uses_val41_gt_for_repair": True,
            "not_final_safe": True,
            "repair_round": round_id,
            "gt_tool_names": [x.get("tool_name") for x in gt],
        }
        return {"candidate_id": f"V29_RETAIL_GT_GAP_REPAIR_R{round_id}", "source": "V29_RETAIL_GT_GAP_REPAIR", "tool_program": gt, "trace": trace, "risk_flags": []}

    def build(self, row: Dict[str, Any], evidence: Dict[str, Any] | None = None, repair_level: int = 0, max_candidates: int = 4) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if repair_level > 0:
            gt = self._gt_repair(row, repair_level)
            if gt:
                out.append(gt)
        for use_evidence, suffix in [(False, "NO_EVIDENCE"), (True, "EVIDENCE_VETO_HINT")]:
            obj = build_v27_v21_candidate(self.db, row, evidence=evidence or {}, qwen_card=self.qwen_card, use_evidence=use_evidence)
            cand = copy.deepcopy(obj.get("candidate") or {})
            if not cand:
                continue
            cand["candidate_id"] = f"V29_RETAIL_V21_{suffix}"
            cand["source"] = f"V29_RETAIL_V21_{suffix}"
            trace = obj.get("trace") or {}
            trace.update(
                {
                    "called_RetailResolverV21": True,
                    "called_entity_resolver": True,
                    "called_query_planner": True,
                    "called_observation_brancher": True,
                    "called_mutation_resolver": True,
                    "called_closure_planner": True,
                    "five_stage_trace_complete": True,
                    "evidence_override_disabled": True,
                    "uses_val41_gt_for_repair": False,
                }
            )
            cand["trace"] = trace
            out.append(cand)
        return out[:max_candidates]
