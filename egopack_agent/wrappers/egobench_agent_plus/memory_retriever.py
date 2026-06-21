# -*- coding: utf-8 -*-
"""V9 memory retrieval for EgoBench Track2 prompts."""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List


CODEX_ROOT = Path(os.environ.get("CODEX_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/codex"))
_MEMORY_DIR_ENV = os.environ.get("TRACK2_MEMORY_BANK_DIR", "memory_bank")
MEMORY_DIR = Path(_MEMORY_DIR_ENV)
if not MEMORY_DIR.is_absolute():
    MEMORY_DIR = CODEX_ROOT / MEMORY_DIR


def _norm_scenario(value: str) -> str:
    value = (value or "").lower()
    for scenario in ("retail", "kitchen", "restaurant", "order"):
        if value.startswith(scenario):
            return scenario
    return value or "global"


def infer_task_type(text: str) -> str:
    low = (text or "").lower()
    checks = [
        ("replace", ["replace", "swap", "instead", "change"]),
        ("remove", ["remove", "delete", "cancel", "take out"]),
        ("payment/tax", ["total", "payment", "pay", "tax", "amount", "cost", "price"]),
        ("nutrition", ["nutrition", "calorie", "protein", "fat", "sugar", "sodium", "carb"]),
        ("recipe/menu/fridge", ["recipe", "ingredient", "fridge", "freezer", "stock", "shopping list", "menu"]),
        ("compare", ["lowest", "highest", "cheapest", "most", "least", "compare"]),
        ("add/remove", ["add", "order", "cart", "include", "put"]),
        ("query", ["what", "which", "whether", "how many", "check", "ask"]),
    ]
    for name, words in checks:
        if any(word in low for word in words):
            return name
    return "general"


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


_CARDS_CACHE: List[Dict[str, Any]] | None = None
_INDEX_CACHE: Dict[str, Any] | None = None


def load_cards() -> List[Dict[str, Any]]:
    global _CARDS_CACHE
    if _CARDS_CACHE is not None:
        return _CARDS_CACHE
    cards = _load_jsonl(MEMORY_DIR / "embeddings" / "cards.jsonl")
    if not cards:
        for name in (
            "scoring_rule_cards.jsonl",
            "failure_pattern_cards.jsonl",
            "success_trajectory_cards.jsonl",
            "canonicalization_cards.jsonl",
            "visual_grounding_cards.jsonl",
        ):
            cards.extend(_load_jsonl(MEMORY_DIR / name))
    if not cards:
        # Graceful fallback: a tiny built-in card keeps V9 runnable before the
        # bank is built.
        cards = [
            {
                "card_id": "fallback::scoring",
                "card_type": "scoring_rule",
                "scenario": "global",
                "task_type": "general",
                "text": "Track2 requires both final result correctness and required tool-process coverage. Use tools before answering when evidence, DB mutation, or aggregate computation is needed.",
            }
        ]
    _CARDS_CACHE = cards
    return cards


def load_index() -> Dict[str, Any]:
    global _INDEX_CACHE
    if _INDEX_CACHE is not None:
        return _INDEX_CACHE
    path = MEMORY_DIR / "embeddings" / "simple_bm25_index.json"
    if path.exists():
        try:
            _INDEX_CACHE = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            _INDEX_CACHE = {}
    else:
        _INDEX_CACHE = {}
    return _INDEX_CACHE


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9_]+", (text or "").lower())


def _candidate_cards(cards: List[Dict[str, Any]], scenario: str, task_type: str) -> List[Dict[str, Any]]:
    scenario = _norm_scenario(scenario)
    allowed_scenarios = {scenario, "global"}
    candidates = [c for c in cards if c.get("scenario", "global") in allowed_scenarios]
    if not candidates:
        candidates = cards
    # Keep task-type specific cards when available, plus general cards.
    typed = [c for c in candidates if c.get("task_type") in {task_type, "general", None}]
    return typed or candidates


def _score_cards(cards: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    index = load_index()
    qtokens = Counter(tokenize(query))
    if not qtokens:
        return cards[:]
    id_to_card = {c.get("card_id"): c for c in cards}
    n_docs = max(1, int(index.get("num_docs") or len(cards) or 1))
    avgdl = float(index.get("avgdl") or 1.0)
    df = index.get("df") or {}
    scored: List[tuple[float, Dict[str, Any]]] = []
    if index.get("docs"):
        allowed = set(id_to_card)
        for doc in index["docs"]:
            cid = doc.get("card_id")
            if cid not in allowed:
                continue
            length = float(doc.get("length") or avgdl or 1.0)
            tf = doc.get("tf") or {}
            score = 0.0
            for tok, qtf in qtokens.items():
                freq = float(tf.get(tok) or 0.0)
                if not freq:
                    continue
                idf = math.log(1.0 + (n_docs - float(df.get(tok, 0)) + 0.5) / (float(df.get(tok, 0)) + 0.5))
                score += idf * (freq * 2.2 / (freq + 1.2 * (1 - 0.75 + 0.75 * length / avgdl))) * qtf
            if score:
                scored.append((score, id_to_card[cid]))
    else:
        for card in cards:
            hay = " ".join([str(card.get(k, "")) for k in ("card_id", "card_type", "scenario", "task_type", "text")])
            toks = Counter(tokenize(hay))
            score = sum(toks[t] * qtf for t, qtf in qtokens.items())
            if score:
                scored.append((float(score), card))
    if not scored:
        return cards[:]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored]


def _choose_by_type(ranked: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    wanted = [
        "scoring_rule",
        "process_template",
        "tool_constitution",
        "failure_pattern",
        "success_trajectory",
    ]
    chosen: List[Dict[str, Any]] = []
    used = set()
    for typ in wanted:
        for card in ranked:
            if card.get("card_type") == typ and card.get("card_id") not in used:
                chosen.append(card)
                used.add(card.get("card_id"))
                break
    if len(chosen) < 3:
        for card in ranked:
            cid = card.get("card_id")
            if cid not in used:
                chosen.append(card)
                used.add(cid)
            if len(chosen) >= 5:
                break
    return chosen[:5]


def _clip(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def format_memory_cards(cards: List[Dict[str, Any]]) -> str:
    if not cards:
        return ""
    labels = {
        "scoring_rule": "Scoring",
        "process_template": "Process",
        "tool_constitution": "Tool constitution",
        "failure_pattern": "Common failure to avoid",
        "success_trajectory": "Similar successful abstract pattern",
        "canonicalization": "Canonicalization",
    }
    lines = ["[Relevant Track2 Memory]"]
    for card in cards:
        label = labels.get(card.get("card_type"), card.get("card_type", "Memory"))
        text = card.get("text") or json.dumps({k: v for k, v in card.items() if k not in {"source_policy"}}, ensure_ascii=False)
        lines.append(f"- {label}: {_clip(text, 420)}")
    return "\n".join(lines)


def retrieve_memory_cards(
    scenario: str,
    user_goal: str = "",
    task_type: str = "",
    stage: str = "",
    latest_observations: str = "",
    limit: int = 5,
) -> List[Dict[str, Any]]:
    scenario = _norm_scenario(scenario)
    task_type = task_type or infer_task_type(" ".join([user_goal, stage, latest_observations]))
    cards = load_cards()
    candidates = _candidate_cards(cards, scenario, task_type)
    query = " ".join([scenario, task_type, user_goal or "", stage or "", latest_observations or ""])
    ranked = _score_cards(candidates, query)
    chosen = _choose_by_type(ranked)
    return chosen[:limit]


def build_memory_prompt(
    scenario: str,
    user_goal: str = "",
    task_type: str = "",
    stage: str = "",
    latest_observations: str = "",
) -> str:
    if os.environ.get("TRACK2_ENABLE_MEMORY_RETRIEVAL", "0") != "1":
        return ""
    cards = retrieve_memory_cards(scenario, user_goal, task_type, stage, latest_observations)
    _record_memory_hits(scenario, cards, user_goal, task_type, stage)
    return format_memory_cards(cards)


def memory_card_ids(cards: Iterable[Dict[str, Any]]) -> List[str]:
    return [str(c.get("card_id")) for c in cards if c.get("card_id")]


def _record_memory_hits(scenario: str, cards: List[Dict[str, Any]], user_goal: str, task_type: str, stage: str) -> None:
    run_id = os.environ.get("TRACK2_RUN_ID") or os.environ.get("TRACK2_OUTPUT_MODEL_NAME") or "manual"
    version = os.environ.get("TRACK2_RUN_VERSION") or "V10_full_memory_final_candidate_draft"
    task_id = os.environ.get("TRACK2_CURRENT_TASK_ID") or "unknown"
    final_eval = os.environ.get("TRACK2_FINAL_EVAL") == "1" or os.environ.get("TRACK2_FINAL_COMPLIANT") == "1"
    out_dir = CODEX_ROOT / "runs" / version / run_id / "memory_hits"
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "task_id": task_id,
        "scenario": scenario,
        "selected_card_ids": memory_card_ids(cards),
        "card_types": [str(c.get("card_type") or "") for c in cards],
        "retrieval_reason": {
            "task_type": task_type,
            "stage": stage,
            "goal_chars": len(user_goal or ""),
            "memory_dir": str(MEMORY_DIR),
        },
        "whether_final_eval": final_eval,
        "no_final_metadata": True,
    }
    with (out_dir / f"{task_id}.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
