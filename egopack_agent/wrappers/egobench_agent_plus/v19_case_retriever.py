#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V19 GT100 case retriever.

This is case retrieval, not memory-card retrieval. It returns complete GT100
program cases from the non-final/non-val41 case library.
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


CODEX = Path(os.environ.get("CODEX_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/codex"))
DEFAULT_CASE_DIR = CODEX / "gt_case_library_v19"

STOP = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "is",
    "are", "you", "your", "my", "please", "could", "would", "ask", "agent",
    "service", "then", "if", "it", "this", "that", "all", "any",
}
AGGREGATE_TOOLS = {
    "compute_total_payment",
    "compute_total_price",
    "compute_total_tax",
    "compute_total_nutrition",
    "compute_total_nutritions",
    "tally_total_nutritional_characteristics",
    "tally_total_tastes",
}
MUTATION_RE = re.compile(r"^(add|remove|delete|update|modify)_|_(to|from)_(cart|order|shopping_list|menu)$")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def tokenize(text: Any) -> set[str]:
    raw = re.findall(r"[A-Za-z0-9_']+", str(text or "").lower())
    return {t for t in raw if len(t) > 2 and t not in STOP}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def scenario_from_spec(spec: str) -> str:
    return re.sub(r"\d+$", "", str(spec or ""))


def classify_task_type(text: str, tool_names: Sequence[str] | None = None) -> str:
    tool_names = list(tool_names or [])
    lower = str(text or "").lower()
    has_mutation = any(MUTATION_RE.search(t) for t in tool_names) or any(x in lower for x in ["add ", "remove ", "delete ", "update ", "cart", "order", "menu", "shopping list"])
    has_agg = any(t in AGGREGATE_TOOLS for t in tool_names) or any(x in lower for x in ["total", "payment", "nutrition", "tax", "summary"])
    has_branch = any(x in lower for x in [" if ", "otherwise", "else", "whether", "if it", "if they"])
    has_rank = any(x in lower for x in ["highest", "lowest", "cheapest", "most", "least", "tie", "tied"])
    has_visual = any(x in lower for x in ["point", "visible", "holding", "shelf", "menu", "bottle", "dish", "video", "image", "left", "right"])
    if has_branch and has_mutation and has_agg:
        return "branch-then-mutation+aggregate"
    if has_mutation and has_agg:
        return "mutation+aggregate"
    if has_mutation:
        return "cart/order/menu mutation"
    if has_agg:
        return "aggregate-required"
    if has_rank:
        return "ranking/filtering"
    if has_visual:
        return "visual-entity query"
    return "query-only"


def program_shape(tool_names: Sequence[str]) -> str:
    parts = []
    for name in tool_names:
        if name in AGGREGATE_TOOLS:
            parts.append("aggregate:" + name)
        elif MUTATION_RE.search(name):
            parts.append("mutation:" + name)
        elif str(name).startswith(("get_", "find_", "filter_", "search_")):
            parts.append("retrieve:" + name)
        else:
            parts.append("other:" + name)
    return " > ".join(parts)


class V19CaseRetriever:
    def __init__(self, case_dir: str | Path | None = None) -> None:
        self.case_dir = Path(case_dir) if case_dir else DEFAULT_CASE_DIR
        self.cases = read_jsonl(self.case_dir / "gt100_cases.jsonl")
        for case in self.cases:
            case.setdefault("_tokens", tokenize(" ".join([
                str(case.get("instruction_text", "")),
                str(case.get("user_utterance_pattern", "")),
                str(case.get("visual_phrase", "")),
            ])))

    def retrieve(self, context: Dict[str, Any], top_k: int = 8) -> List[Dict[str, Any]]:
        scenario = str(context.get("scenario") or scenario_from_spec(context.get("spec", "")))
        task_type = context.get("task_type") or classify_task_type(context.get("instruction", ""))
        text_tokens = tokenize(" ".join([
            str(context.get("instruction", "")),
            str(context.get("dialogue_history", "")),
            str(context.get("visual_text", "")),
            str(context.get("intent", "")),
        ]))
        visual_tokens = tokenize(context.get("visual_text", ""))
        wanted_entities = set(context.get("entity_types") or [])
        scored = []
        for case in self.cases:
            if not case.get("excluded_final309") or not case.get("excluded_val41"):
                continue
            score = 0.0
            reasons = []
            if case.get("scenario") == scenario:
                score += 5.0
                reasons.append("scenario_exact")
            elif case.get("scenario") == scenario_from_spec(context.get("spec", "")):
                score += 3.0
                reasons.append("scenario_from_spec")
            if case.get("task_type") == task_type:
                score += 2.0
                reasons.append("task_type_match")
            sim = jaccard(text_tokens, case.get("_tokens", set()))
            score += 3.0 * sim
            if sim:
                reasons.append(f"utterance_jaccard={sim:.3f}")
            vsim = jaccard(visual_tokens, tokenize(case.get("visual_phrase", "")))
            score += 1.5 * vsim
            if vsim:
                reasons.append(f"visual_jaccard={vsim:.3f}")
            case_entities = set(case.get("entity_types") or [])
            if wanted_entities and case_entities:
                compat = len(wanted_entities & case_entities) / max(1, len(wanted_entities | case_entities))
                score += compat
                if compat:
                    reasons.append(f"entity_compat={compat:.3f}")
            current_shape = context.get("program_shape") or ""
            if current_shape and current_shape == case.get("program_shape"):
                score += 1.0
                reasons.append("program_shape_exact")
            # Prefer compact programs when scores tie.
            score -= 0.01 * len(case.get("tool_program") or [])
            scored.append({
                "case": case,
                "score": score,
                "reasons": reasons or ["weak_lexical_match"],
                "avoid_same_val41_final_source": True,
            })
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]


def retrieve_cases(context: Dict[str, Any], top_k: int = 8, case_dir: str | Path | None = None) -> List[Dict[str, Any]]:
    return V19CaseRetriever(case_dir).retrieve(context, top_k=top_k)
