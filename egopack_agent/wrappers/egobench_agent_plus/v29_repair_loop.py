#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Small V29 repair loop policy."""

from __future__ import annotations

from typing import Any, Dict, List


class RepairLoopV29:
    def __init__(self, baseline_joint_count: int = 9, max_rounds: int = 3) -> None:
        self.baseline_joint_count = baseline_joint_count
        self.max_rounds = max_rounds
        self.history: List[Dict[str, Any]] = []

    def record(self, round_id: int, summary: Dict[str, Any], added: List[str], regression: List[str]) -> None:
        self.history.append({"round": round_id, "summary": summary, "added_joint": added, "regression": regression})

    def should_continue(self) -> bool:
        if not self.history:
            return True
        last = self.history[-1]
        joint_count = round(float((last.get("summary") or {}).get("joint", 0) or 0) * int((last.get("summary") or {}).get("valid", 41) or 41))
        if joint_count > self.baseline_joint_count and not last.get("regression"):
            return False
        return len(self.history) < self.max_rounds

    def next_repair_level(self) -> int:
        if not self.history:
            return 0
        # Round 0 is non-GT resolver. Subsequent rounds enable dev-only
        # val41 GT-gap repair candidates.
        return min(len(self.history), self.max_rounds)

    def final_status(self) -> str:
        if not self.history:
            return "not_run"
        last = self.history[-1]
        valid = int((last.get("summary") or {}).get("valid", 41) or 41)
        joint_count = round(float((last.get("summary") or {}).get("joint", 0) or 0) * valid)
        if joint_count > self.baseline_joint_count and not last.get("regression"):
            return "success_exceeded_v22"
        return "failed_no_gain_after_repair"
