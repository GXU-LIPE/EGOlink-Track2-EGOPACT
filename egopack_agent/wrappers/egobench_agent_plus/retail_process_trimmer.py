# -*- coding: utf-8 -*-
"""V10 retail process-aware trimming hints."""
from __future__ import annotations

import os
from typing import Any, Dict


def enabled() -> bool:
    return os.environ.get("TRACK2_ENABLE_RETAIL_PROCESS_TRIMMER", "0") == "1" or os.environ.get("TRACK2_ENABLE_RETAIL_CANDIDATE_NARROWER", "0") == "1" or os.environ.get("TRACK2_ENABLE_RETAIL_NARROWER", "0") == "1"


def prompt() -> str:
    if not enabled():
        return ""
    return "\n".join([
        "[V10 Retail Process-Aware Trimming]",
        "- For visual/filter retail tasks, narrow candidates before numeric attribute calls.",
        "- Suggested filter order: category -> country/origin -> brand/name/visual clue -> taste/profile -> nutrition/price/discount/tax.",
        "- If candidate count is above five, do not query price/tax/discount/nutrition for the whole catalog.",
        "- After 25 tool calls, narrow-only. After 40, block new broad retrieval. After 60, choose the most grounded candidate and finish required process.",
        "- Lowest/cheapest/highest/healthiest is a ranking objective, not an automatic final aggregate request.",
    ])


def process_card() -> Dict[str, Any]:
    return {
        "card_id": "v10::retail_process_trimming",
        "card_type": "process_template",
        "scenario": "retail",
        "task_type": "visual/filter",
        "text": "Retail process: narrow by category/origin/brand/name/taste before numeric attributes; avoid catalog-wide price/tax/discount/nutrition sweeps; ranking words are filters, not automatic aggregate triggers.",
        "no_final_metadata": True,
    }
