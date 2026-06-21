#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Grounded best-first tool search for V33."""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List

from .v32_tool_loop_guard import execute_calls


MUTATION_RE = re.compile(r"^(add|remove|delete|update|modify)_|_(to|from)_(cart|order|shopping_list|menu)$")
QUERY_PREFIXES = ("get_", "find_", "filter_", "list_")
AGG_TOOLS = {
    "compute_total_payment",
    "compute_total_tax",
    "compute_total_nutrition",
    "compute_total_nutritions",
    "tally_total_nutritional_characteristics",
    "tally_total_tastes",
    "get_user_order_summary",
}


def norm_text(v: Any) -> str:
    return re.sub(r"\s+", " ", str(v or "").strip().lower())


def call_key(call: Dict[str, Any]) -> str:
    return json.dumps({"tool_name": call.get("tool_name"), "parameters": call.get("parameters") or {}}, ensure_ascii=False, sort_keys=True)


@dataclass
class ActionProposal:
    source: str
    action: Dict[str, Any]
    score: float = 0.0
    reason: str = ""
    api_error: str = ""


@dataclass
class SearchNode:
    task_key: str
    scenario: str
    db: Any
    tool_history: List[Dict[str, Any]] = field(default_factory=list)
    observations: List[Dict[str, Any]] = field(default_factory=list)
    pinned: Dict[str, str] = field(default_factory=dict)
    branch_facts: List[str] = field(default_factory=list)
    mutation_done: bool = False
    closure_done: List[str] = field(default_factory=list)
    score: float = 0.0
    risk_flags: List[str] = field(default_factory=list)
    depth: int = 0
    proposer_sources: List[str] = field(default_factory=list)
    verifier_trace: Dict[str, Any] = field(default_factory=dict)

    def clone_with(self, proposal: ActionProposal, result_rows: List[Dict[str, Any]], verifier_score: float, verifier_trace: Dict[str, Any]) -> "SearchNode":
        action = proposal.action
        names = [r.get("tool_name") for r in result_rows]
        obs = self.observations + result_rows
        history = self.tool_history + [action]
        closure = list(self.closure_done)
        mutation = self.mutation_done or bool(MUTATION_RE.search(action.get("tool_name", "")))
        if action.get("tool_name") in AGG_TOOLS:
            closure.append(action.get("tool_name", ""))
        pinned = dict(self.pinned)
        for k, v in (action.get("parameters") or {}).items():
            if k in {"user_id", "restaurant_name", "product_name", "dish_name", "set_meal_name", "ingredient_name", "recipe_name"}:
                pinned[k] = str(v)
        flags = list(dict.fromkeys(self.risk_flags + verifier_trace.get("risk_flags", [])))
        return SearchNode(
            task_key=self.task_key,
            scenario=self.scenario,
            db=copy.deepcopy(self.db),
            tool_history=history,
            observations=obs,
            pinned=pinned,
            branch_facts=self.branch_facts + verifier_trace.get("branch_facts", []),
            mutation_done=mutation,
            closure_done=list(dict.fromkeys(closure)),
            score=self.score + proposal.score + verifier_score,
            risk_flags=flags,
            depth=self.depth + 1,
            proposer_sources=self.proposer_sources + [proposal.source],
            verifier_trace=verifier_trace | {"last_result_tools": names},
        )

    def export(self) -> Dict[str, Any]:
        return {
            "task_key": self.task_key,
            "scenario": self.scenario,
            "tool_history": self.tool_history,
            "observations": self.observations[-12:],
            "pinned": self.pinned,
            "branch_facts": self.branch_facts,
            "mutation_done": self.mutation_done,
            "closure_done": self.closure_done,
            "score": self.score,
            "risk_flags": self.risk_flags,
            "depth": self.depth,
            "proposer_sources": self.proposer_sources,
            "verifier_trace": self.verifier_trace,
        }


class NoGTVerifier:
    def __init__(self, instruction: str, scenario: str, evidence: Dict[str, Any] | None = None) -> None:
        self.instruction = instruction or ""
        self.scenario = scenario
        self.evidence = evidence or {}
        self.text = norm_text(instruction)

    def required_closure(self) -> List[str]:
        out: List[str] = []
        if "tax" in self.text:
            out.append("compute_total_tax")
        if any(x in self.text for x in ("payment", "payable", "total cost", "amount due")):
            out.append("compute_total_payment")
        if any(x in self.text for x in ("nutrition", "nutritional", "protein", "fat", "sugar", "calcium")):
            out.extend(["compute_total_nutrition", "compute_total_nutritions", "tally_total_nutritional_characteristics"])
        if "summary" in self.text and self.scenario in {"order", "restaurant"}:
            out.append("get_user_order_summary")
        return out

    def mutation_intent(self) -> bool:
        return any(x in self.text for x in ("add", "remove", "delete", "update", "cart", "order", "shopping list", "menu"))

    def score_action(self, node: SearchNode, action: Dict[str, Any], results: List[Dict[str, Any]]) -> tuple[float, Dict[str, Any]]:
        name = action.get("tool_name", "")
        params = action.get("parameters") or {}
        risk: List[str] = []
        plus: List[str] = []
        score = 0.0
        if not name or any(v in ("", None, [], {}) for v in params.values()):
            score -= 5.0
            risk.append("empty_params")
        if name.startswith(QUERY_PREFIXES):
            score += 1.5
            plus.append("query_action")
            if any(k.endswith("_name") or k == "user_id" for k in params):
                score += 1.0
                plus.append("entity_specific_query")
        if MUTATION_RE.search(name):
            if not node.tool_history:
                score -= 2.0
                risk.append("mutation_first")
            if not node.observations:
                score -= 1.3
                risk.append("mutation_without_observation")
            if any(k.endswith("_name") for k in params):
                score += 1.0
                plus.append("mutation_target_named")
            score += 1.2
            plus.append("mutation_action")
        if name in AGG_TOOLS:
            if node.mutation_done or any("summary" in self.text for _ in [0]):
                score += 2.0
                plus.append("closure_action")
            else:
                score -= 0.8
                risk.append("early_closure")
        if name.startswith(("find_products_by_price_range", "get_all_", "list_all_")) and len(node.tool_history) < 2:
            score -= 3.0
            risk.append("leading_broad_scan")
        if any((r.get("status") == "error") for r in results):
            score -= 2.0
            risk.append("tool_error")
        else:
            score += 0.5
        if self.mutation_intent() and node.mutation_done:
            score += 0.5
        req = self.required_closure()
        if req and any(x in [name] + node.closure_done for x in req):
            score += 1.2
            plus.append("requested_closure_progress")
        if req and node.mutation_done and name not in AGG_TOOLS and len(node.tool_history) >= 6:
            score -= 0.8
            risk.append("closure_still_missing")
        if call_key(action) in {call_key(x) for x in node.tool_history}:
            score -= 2.0
            risk.append("repeated_action")
        trace = {"risk_flags": risk, "positive": plus, "branch_facts": [], "verifier_delta": score}
        return score, trace

    def terminal_score(self, node: SearchNode) -> float:
        score = node.score
        if self.mutation_intent() and not node.mutation_done:
            score -= 3.0
        req = self.required_closure()
        if req:
            if any(x in node.closure_done for x in req):
                score += 2.0
            else:
                score -= 2.0
        if node.depth == 0:
            score -= 5.0
        score -= max(0, node.depth - 12) * 0.3
        return score

    def reliable_for_takeover(self, node: SearchNode) -> bool:
        if "tool_error" in node.risk_flags or "empty_params" in node.risk_flags:
            return False
        if "leading_broad_scan" in node.risk_flags:
            return False
        if self.mutation_intent() and not node.mutation_done:
            return False
        req = self.required_closure()
        if req and not any(x in node.closure_done for x in req):
            return False
        return self.terminal_score(node) >= 4.0 and len(node.tool_history) > 0


class BeamSearchRunner:
    def __init__(
        self,
        *,
        proposers: List[Any],
        verifier: NoGTVerifier,
        beam_size: int = 5,
        max_depth: int = 12,
        max_actions_per_step: int = 6,
        max_trajectories: int = 30,
    ) -> None:
        self.proposers = proposers
        self.verifier = verifier
        self.beam_size = beam_size
        self.max_depth = max_depth
        self.max_actions_per_step = max_actions_per_step
        self.max_trajectories = max_trajectories
        self.action_logs: List[Dict[str, Any]] = []
        self.node_logs: List[Dict[str, Any]] = []
        self.verifier_logs: List[Dict[str, Any]] = []

    def run(self, root: SearchNode, ctx: Dict[str, Any]) -> List[SearchNode]:
        beam = [root]
        completed: List[SearchNode] = []
        seen_programs = set()
        for depth in range(self.max_depth):
            next_nodes: List[SearchNode] = []
            for rank, node in enumerate(beam):
                ctx = dict(ctx)
                ctx["node_rank"] = rank
                proposals: List[ActionProposal] = []
                for proposer in self.proposers:
                    raw = proposer.propose(ctx, node, limit=self.max_actions_per_step)
                    for p in raw:
                        action = p.get("action") or {}
                        if not action:
                            self.action_logs.append({"task_key": node.task_key, "depth": depth, "source": p.get("source"), "api_error": p.get("api_error"), "rejected": True})
                            continue
                        proposals.append(ActionProposal(source=p.get("source", proposer.__class__.__name__), action=action, score=float(p.get("score", 0)), reason=p.get("reason", ""), api_error=p.get("api_error", "")))
                dedup: Dict[str, ActionProposal] = {}
                for p in proposals:
                    key = call_key(p.action)
                    if key not in dedup or p.score > dedup[key].score:
                        dedup[key] = p
                proposals = sorted(dedup.values(), key=lambda p: p.score, reverse=True)[: self.max_actions_per_step]
                self.action_logs.extend(
                    {
                        "task_key": node.task_key,
                        "depth": depth,
                        "source": p.source,
                        "action": p.action,
                        "proposal_score": p.score,
                        "reason": p.reason,
                    }
                    for p in proposals
                )
                for p in proposals:
                    child_db = copy.deepcopy(node.db)
                    results = execute_calls(child_db, [p.action])
                    delta, trace = self.verifier.score_action(node, p.action, results)
                    child = node.clone_with(p, results, delta, trace)
                    child.db = child_db
                    pkey = json.dumps([call_key(x) for x in child.tool_history], ensure_ascii=False)
                    if pkey in seen_programs:
                        continue
                    seen_programs.add(pkey)
                    next_nodes.append(child)
                    self.verifier_logs.append({"task_key": node.task_key, "depth": depth, "action": p.action, "delta": delta, "trace": trace})
                    if self.verifier.reliable_for_takeover(child):
                        completed.append(child)
                    if len(completed) >= self.max_trajectories:
                        break
                if len(completed) >= self.max_trajectories:
                    break
            if not next_nodes:
                break
            beam = sorted(next_nodes, key=lambda n: self.verifier.terminal_score(n), reverse=True)[: self.beam_size]
            self.node_logs.extend(n.export() for n in beam)
            if len(completed) >= self.max_trajectories:
                break
        pool = completed + beam
        pool = sorted(pool, key=lambda n: self.verifier.terminal_score(n), reverse=True)
        return pool[: self.max_trajectories]
