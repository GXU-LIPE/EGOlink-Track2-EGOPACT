#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate V19 trajectory-reuse program candidates."""

from __future__ import annotations

import copy
from collections import Counter, defaultdict
from typing import Any, Dict, List

from .v19_case_retriever import program_shape, retrieve_cases
from .v19_nonoracle_program_scorer import rank_candidates
from .v19_slot_rewriter import rewrite_case_program


def normalize_program(item: Dict[str, Any] | None) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not isinstance(item, dict):
        return out
    for entry in item.get("tool_calls", []) or []:
        calls = entry.get("calls") if isinstance(entry, dict) else None
        if isinstance(calls, list):
            for call in calls:
                if isinstance(call, dict) and call.get("tool_name"):
                    out.append({
                        "tool_name": call.get("tool_name"),
                        "parameters": copy.deepcopy(call.get("parameters", {})),
                        "stage": "baseline",
                    })
    return out


def make_baseline_candidate(source: str, item: Dict[str, Any] | None) -> Dict[str, Any]:
    program = normalize_program(item)
    return {
        "candidate_id": source,
        "source": source,
        "source_case_ids": [],
        "tool_program": program,
        "slot_confidence": {},
        "risk_flags": [] if program else ["empty_program"],
        "expected_closure": {"from_baseline": source},
        "program_shape_confidence": 0.5 if program else 0.0,
    }


def consensus_program(cases: List[Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
    if not cases:
        return {"tool_program": [], "trace": {}, "slot_confidence": {}, "risk_flags": ["no_cases"]}
    seqs = [tuple(c["case"].get("tool_name_sequence") or []) for c in cases]
    if not seqs:
        chosen = cases[0]["case"]
    else:
        common = Counter(seqs).most_common(1)[0][0]
        chosen = next((c["case"] for c in cases if tuple(c["case"].get("tool_name_sequence") or []) == common), cases[0]["case"])
    return rewrite_case_program(chosen, context)


def generate_candidates(context: Dict[str, Any], v10_item: Dict[str, Any] | None = None, v14_item: Dict[str, Any] | None = None, top_k: int = 8) -> Dict[str, Any]:
    hits = retrieve_cases(context, top_k=top_k)
    candidates: List[Dict[str, Any]] = []
    traces = {"case_hit_trace": [], "slot_rewrite_trace": []}
    for h in hits:
        case = h["case"]
        traces["case_hit_trace"].append({
            "case_id": case.get("case_id"),
            "score": h.get("score"),
            "reasons": h.get("reasons"),
            "spec": case.get("spec"),
            "program_shape": case.get("program_shape"),
        })
    if hits:
        rewritten = rewrite_case_program(hits[0]["case"], context)
        traces["slot_rewrite_trace"].append(rewritten["trace"])
        candidates.append({
            "candidate_id": "case_top1_rewrite",
            "source": "case_top1",
            "source_case_ids": [hits[0]["case"].get("case_id")],
            "tool_program": rewritten["tool_program"],
            "slot_confidence": rewritten["slot_confidence"],
            "risk_flags": rewritten["risk_flags"],
            "expected_closure": {"source": "top1"},
            "program_shape_confidence": max(0.0, min(1.0, hits[0].get("score", 0) / 10.0)),
        })
    top3 = hits[:3]
    if top3:
        rewritten = consensus_program(top3, context)
        traces["slot_rewrite_trace"].append(rewritten.get("trace", {}))
        candidates.append({
            "candidate_id": "case_top3_vote",
            "source": "case_top3",
            "source_case_ids": [h["case"].get("case_id") for h in top3],
            "tool_program": rewritten["tool_program"],
            "slot_confidence": rewritten["slot_confidence"],
            "risk_flags": rewritten["risk_flags"],
            "expected_closure": {"source": "top3_consensus"},
            "program_shape_confidence": 0.7,
        })
    by_shape: Dict[str, Dict[str, Any]] = {}
    for h in hits:
        shape = h["case"].get("program_shape") or program_shape(h["case"].get("tool_name_sequence") or [])
        if shape not in by_shape:
            by_shape[shape] = h
    for idx, h in enumerate(list(by_shape.values())[:3], start=1):
        rewritten = rewrite_case_program(h["case"], context)
        traces["slot_rewrite_trace"].append(rewritten["trace"])
        candidates.append({
            "candidate_id": f"case_diverse_programs_{idx}",
            "source": "diverse",
            "source_case_ids": [h["case"].get("case_id")],
            "tool_program": rewritten["tool_program"],
            "slot_confidence": rewritten["slot_confidence"],
            "risk_flags": rewritten["risk_flags"],
            "expected_closure": {"source": "diverse_shape"},
            "program_shape_confidence": max(0.0, min(1.0, h.get("score", 0) / 10.0)),
        })
    candidates.append(make_baseline_candidate("v14", v14_item))
    candidates.append(make_baseline_candidate("v10", v10_item))
    # Simple repair: prefer top case but append baseline aggregate if top case lacks one.
    if candidates and v14_item:
        base = copy.deepcopy(candidates[0])
        v14_prog = normalize_program(v14_item)
        names = {x.get("tool_name") for x in base.get("tool_program") or []}
        appended = []
        for step in v14_prog:
            name = step.get("tool_name")
            if name and name.startswith(("compute_", "tally_")) and name not in names:
                appended.append(step)
        if appended:
            base["candidate_id"] = "repair_program"
            base["source"] = "repair"
            base["tool_program"] = (base.get("tool_program") or []) + appended
            base.setdefault("risk_flags", []).append("baseline_aggregate_repair")
            base["expected_closure"] = {"appended": [x.get("tool_name") for x in appended]}
            candidates.append(base)
    ranked = rank_candidates(candidates, context)
    return {"candidates": candidates, "ranked": ranked, "traces": traces}
