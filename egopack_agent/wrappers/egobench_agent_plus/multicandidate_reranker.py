# -*- coding: utf-8 -*-
"""Compatibility wrapper for deterministic candidate reranking."""
from __future__ import annotations

from typing import Any, Dict, List

from .v9_multicandidate_reranker import score_candidate as _score_candidate
from .v9_multicandidate_reranker import select_candidate as _select_candidate


def score_candidate(candidate: Any, scenario: str, state: Dict[str, Any]) -> Dict[str, Any]:
    return _score_candidate(candidate, scenario, state, turn=0)


def select_candidate(candidates: List[Any], scenario: str, state: Dict[str, Any], turn: int) -> Dict[str, Any]:
    return _select_candidate(candidates, scenario, state, turn)
