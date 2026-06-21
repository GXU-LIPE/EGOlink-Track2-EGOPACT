#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V31 LOO candidate generation and dry-run execution."""

from __future__ import annotations

import copy
from typing import Any, Dict, List

from .v24_candidate_dryrun_and_selector import dryrun_program
from .v30_prior_retrieval_agent import materialize_program
from .v31_loo_slot_filler import DetaskedSlotFillerV31


class V31LOOProgramExecutor:
    def __init__(self, slot_filler: DetaskedSlotFillerV31) -> None:
        self.slot_filler = slot_filler

    def build_candidates(
        self,
        task_key: str,
        scenario: str,
        row: Dict[str, Any],
        db: Any,
        loo_cases: List[Dict[str, Any]],
        v22_item: Dict[str, Any] | None,
        slot_only_candidates: List[Dict[str, Any]],
        heldout_case_id: str = "",
        top_k: int = 8,
    ) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        if v22_item:
            candidates.append(
                {
                    "candidate_id": "V22_PROTECTED_FALLBACK",
                    "source": "V22",
                    "tool_program": self._item_program(v22_item),
                    "slot_set_id": "",
                    "risk_flags": ["fallback"],
                }
            )
        for cand in slot_only_candidates[:2]:
            row_cand = copy.deepcopy(cand)
            row_cand["candidate_id"] = "V30_SLOT_ONLY_" + str(row_cand.get("candidate_id", ""))[:120]
            row_cand["source"] = "V30_SLOT_ONLY"
            candidates.append(row_cand)

        retrieved = self.slot_filler.retrieve_cases(loo_cases, scenario, row.get("Instruction", ""), heldout_case_id, k=top_k)
        for rank, case in enumerate(retrieved[:top_k], 1):
            abstract = case.get("abstract_tool_chain") or []
            slot_sets = self.slot_filler.fill_slot_sets(task_key, scenario, row, db, abstract, case, max_sets=3)
            for slot_set in slot_sets:
                program = materialize_program(abstract, slot_set.get("placeholder_values") or {})
                candidates.append(
                    {
                        "candidate_id": f"V31_LOO_{rank}_{slot_set.get('slot_set_id')}_{case.get('program_family')}"[:180],
                        "source": "V31_LOO_SLOT_FILLER",
                        "tool_program": program,
                        "prior_id": case.get("prior_id"),
                        "prior_case_id": case.get("case_id"),
                        "program_family": case.get("program_family"),
                        "retrieval_score": case.get("retrieval_score", 0.0),
                        "slot_set_id": slot_set.get("slot_set_id"),
                        "slot_values_used": slot_set.get("placeholder_values"),
                        "slot_evidence": slot_set.get("evidence"),
                        "slot_confidence": slot_set.get("confidence"),
                        "risk_flags": slot_set.get("risk_flags", []),
                        "heldout_excluded": True,
                    }
                )
        # Conservative closure repair: if a V31 candidate has the right mutation
        # prefix but lacks an aggregate present in a slot-only/V22 candidate,
        # append that aggregate shape with current slot values when possible.
        for base in list(candidates):
            if base.get("source") != "V31_LOO_SLOT_FILLER":
                continue
            repaired = self._closure_repair(base, candidates)
            if repaired:
                candidates.append(repaired)
        return candidates[:40]

    def dryrun_candidates(self, scenario: str, db_factory: Any, row: Dict[str, Any], candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for cand in candidates:
            c = copy.deepcopy(cand)
            c["dryrun"] = dryrun_program(scenario, db_factory(), c.get("tool_program") or [], row.get("Instruction", ""))
            out.append(c)
        return out

    @staticmethod
    def _item_program(item: Dict[str, Any]) -> List[Dict[str, Any]]:
        calls: List[Dict[str, Any]] = []
        for block in item.get("tool_calls") or []:
            for call in block.get("calls") or []:
                calls.append({"tool_name": call.get("tool_name") or call.get("name"), "parameters": call.get("parameters") or {}})
        return calls

    @staticmethod
    def _closure_repair(base: Dict[str, Any], candidates: List[Dict[str, Any]]) -> Dict[str, Any] | None:
        program = base.get("tool_program") or []
        names = [c.get("tool_name") for c in program]
        if any(str(n).startswith(("compute_", "tally_", "get_user_order_summary")) for n in names):
            return None
        for other in candidates:
            for call in other.get("tool_program") or []:
                tool = str(call.get("tool_name") or "")
                if tool.startswith(("compute_total_", "tally_total_", "get_user_order_summary")):
                    repaired = copy.deepcopy(base)
                    repaired["candidate_id"] = str(base.get("candidate_id", "V31"))[:150] + "_CLOSURE_REPAIR"
                    repaired["source"] = "V31_CLOSURE_REPAIR"
                    repaired["tool_program"] = copy.deepcopy(program) + [copy.deepcopy(call)]
                    repaired["risk_flags"] = sorted(set((repaired.get("risk_flags") or []) + ["closure_repair_from_candidate_shape"]))
                    return repaired
        return None
