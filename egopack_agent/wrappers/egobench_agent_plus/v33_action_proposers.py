#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Action proposers for V33 grounded tool search.

The proposers return next actions only.  They do not inspect GT and do not
copy final metadata.  Cached V32/V30 artifacts are used only as non-GT
proposal sources.
"""

from __future__ import annotations

import inspect
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .v25_evidence_entity_matcher import compact_db_entity_list, norm_text
from .v32_native_vision_service_agent import call_gpt55
from .v32_tool_loop_guard import normalize_tool_calls


CODEX = Path(os.environ.get("CODEX_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/codex"))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _call_key(call: Dict[str, Any]) -> str:
    return json.dumps({"tool_name": call.get("tool_name"), "parameters": call.get("parameters") or {}}, ensure_ascii=False, sort_keys=True)


def _filter_params(db: Any, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    if not hasattr(db, tool_name):
        return params or {}
    try:
        sig = inspect.signature(getattr(db, tool_name))
        return {k: v for k, v in (params or {}).items() if k in sig.parameters}
    except Exception:
        return params or {}


def _valid_action(db: Any, action: Dict[str, Any]) -> Dict[str, Any] | None:
    name = action.get("tool_name") or action.get("name")
    if not name or not hasattr(db, str(name)):
        return None
    params = _filter_params(db, str(name), action.get("parameters") or action.get("arguments") or {})
    try:
        sig = inspect.signature(getattr(db, str(name)))
        missing = [
            k for k, v in sig.parameters.items()
            if v.default is inspect.Parameter.empty and k not in params
        ]
        if missing:
            return None
    except Exception:
        pass
    if any(v in ("", None, [], {}) for v in params.values()):
        return None
    return {"tool_name": str(name), "parameters": params}


def _extract_user_id(text: str) -> str:
    m = re.search(r"user id is\s*([A-Za-z0-9_\-]+)|user id[:：]\s*([A-Za-z0-9_\-]+)", text or "", flags=re.I)
    return next((g for g in (m.groups() if m else []) if g), "")


def _slot_candidates(evidence: Dict[str, Any], scenario: str, db: Any) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    matches = evidence.get("canonical_matches") if isinstance(evidence, dict) else {}
    if isinstance(matches, dict):
        for typ, vals in matches.items():
            names = []
            for v in vals or []:
                if isinstance(v, dict) and v.get("canonical_name"):
                    names.append(str(v["canonical_name"]))
            if names:
                out[typ] = list(dict.fromkeys(names))[:8]
    dbs = compact_db_entity_list(scenario, db, limit_per_type=80)
    for typ, vals in dbs.items():
        out.setdefault(typ, vals[:8])
    return out


class CachedTrajectoryProposer:
    def __init__(self, records_path: Path, source_name: str) -> None:
        self.source_name = source_name
        self.by_key: Dict[str, List[List[Dict[str, Any]]]] = {}
        for row in _read_jsonl(records_path):
            if row.get("api_errors"):
                continue
            key = row.get("task_key")
            prog = row.get("tool_program") or []
            if key and prog:
                self.by_key.setdefault(str(key), []).append(prog)

    def propose(self, ctx: Dict[str, Any], node: Any, limit: int = 6) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        prefix_len = len(node.tool_history)
        seen = set()
        for prog in self.by_key.get(ctx["task_key"], []):
            if prefix_len < len(prog):
                action = _valid_action(node.db, prog[prefix_len])
                if action and _call_key(action) not in seen:
                    seen.add(_call_key(action))
                    out.append({"source": self.source_name, "action": action, "score": 2.0, "reason": f"cached prefix step {prefix_len}"})
            if len(out) >= limit:
                break
        return out


class ProgramPriorProposer:
    def __init__(self, path: Path | None = None) -> None:
        self.by_key: Dict[str, List[List[Dict[str, Any]]]] = {}
        path = path or (CODEX / "analysis" / "v30_prior_candidate_programs.jsonl")
        for row in _read_jsonl(path):
            if "dev_only_prior" in (row.get("risk_flags") or []):
                continue
            tr = row.get("trace") or {}
            if tr.get("uses_val41_gt_experience"):
                continue
            key = row.get("task_key")
            prog = row.get("tool_program") or (row.get("candidate") or {}).get("tool_program") or []
            if key and prog:
                self.by_key.setdefault(str(key), []).append(prog)

    def propose(self, ctx: Dict[str, Any], node: Any, limit: int = 6) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        seen = set()
        prefix_len = len(node.tool_history)
        for prog in self.by_key.get(ctx["task_key"], []):
            if prefix_len < len(prog):
                action = _valid_action(node.db, prog[prefix_len])
                if action and _call_key(action) not in seen:
                    seen.add(_call_key(action))
                    out.append({"source": "ProgramPriorProposer", "action": action, "score": 1.2, "reason": "V30 slot-level prior prefix"})
            if len(out) >= limit:
                break
        return out


class V21RetailProposer:
    def propose(self, ctx: Dict[str, Any], node: Any, limit: int = 6) -> List[Dict[str, Any]]:
        if ctx.get("scenario") != "retail":
            return []
        try:
            from .v21_retail_resolver import RetailResolverV21
            resolver = RetailResolverV21(node.db)
            built = resolver.build({"Instruction": ctx.get("instruction", "")})
            prog = ((built.get("candidate") or {}).get("tool_program") or [])
        except Exception:
            return []
        idx = len(node.tool_history)
        if idx >= len(prog):
            return []
        action = _valid_action(node.db, prog[idx])
        if not action:
            return []
        return [{"source": "V21RetailProposer", "action": action, "score": 1.8, "reason": "V21 retail observation-driven prefix"}]


class DeterministicLegalActionProposer:
    QUERY_PREFIXES = ("get_", "find_", "filter_", "list_")
    CLOSURES = {
        "compute_total_payment",
        "compute_total_tax",
        "compute_total_nutrition",
        "compute_total_nutritions",
        "tally_total_nutritional_characteristics",
        "tally_total_tastes",
        "get_user_order_summary",
    }

    def propose(self, ctx: Dict[str, Any], node: Any, limit: int = 6) -> List[Dict[str, Any]]:
        text = ctx.get("instruction", "")
        slots = _slot_candidates(ctx.get("evidence") or {}, ctx.get("scenario", ""), node.db)
        user_id = _extract_user_id(text)
        out: List[Dict[str, Any]] = []
        names = [n for n in dir(node.db) if not n.startswith("_") and callable(getattr(node.db, n))]

        def add(tool: str, params: Dict[str, Any], score: float, reason: str) -> None:
            if len(out) >= limit:
                return
            action = _valid_action(node.db, {"tool_name": tool, "parameters": params})
            if action and _call_key(action) not in {_call_key(x["action"]) for x in out}:
                out.append({"source": "DeterministicLegalActionProposer", "action": action, "score": score, "reason": reason})

        # Prefer closure once mutation has happened.
        if node.mutation_done:
            lower = norm_text(text)
            for tool in names:
                if tool not in self.CLOSURES:
                    continue
                if "payment" in tool and not any(x in lower for x in ("payment", "payable", "cost", "price")):
                    continue
                if "tax" in tool and "tax" not in lower:
                    continue
                if "nutrition" in tool and not any(x in lower for x in ("nutrition", "nutritional", "protein", "fat", "sugar", "calcium")):
                    continue
                params = {"user_id": user_id} if user_id else {}
                add(tool, params, 2.5, "required closure after mutation")
            if out:
                return out[:limit]

        # Entity-specific queries before mutation.
        entity_map = {
            "product_name": slots.get("product") or slots.get("primary_product") or [],
            "dish_name": slots.get("dish") or [],
            "set_meal_name": slots.get("set_meal") or [],
            "restaurant_name": slots.get("restaurant") or [],
            "ingredient_name": slots.get("ingredient") or [],
            "recipe_name": slots.get("recipe") or [],
            "category": slots.get("category") or [],
            "user_id": [user_id] if user_id else [],
        }
        for tool in names:
            if not tool.startswith(self.QUERY_PREFIXES):
                continue
            try:
                sig = inspect.signature(getattr(node.db, tool))
            except Exception:
                continue
            params: Dict[str, Any] = {}
            ok = True
            for p in sig.parameters:
                vals = entity_map.get(p)
                if vals:
                    params[p] = vals[0]
                else:
                    ok = False
                    break
            if ok:
                add(tool, params, 1.4, "entity-specific legal query")
            if len(out) >= limit:
                break

        # Minimal mutations when there is current DB evidence and user intent.
        lower = norm_text(text)
        if any(x in lower for x in ("add", "remove", "delete", "update")) and user_id:
            for tool in names:
                if not re.search(r"^(add|remove|delete|update|modify)_|_(to|from)_(cart|order|shopping_list|menu)$", tool):
                    continue
                try:
                    sig = inspect.signature(getattr(node.db, tool))
                except Exception:
                    continue
                params = {}
                ok = True
                for p in sig.parameters:
                    vals = entity_map.get(p)
                    if vals:
                        params[p] = vals[0]
                    elif p in {"quantity", "amount"}:
                        params[p] = 1
                    else:
                        ok = False
                        break
                if ok:
                    add(tool, params, 1.0, "legal mutation from current evidence")
                if len(out) >= limit:
                    break
        return out[:limit]


class V32NativeVisionProposer:
    def __init__(self, live_depth: int = 4, live_nodes_per_depth: int = 2) -> None:
        self.live_depth = live_depth
        self.live_nodes_per_depth = live_nodes_per_depth

    def propose(self, ctx: Dict[str, Any], node: Any, limit: int = 6) -> List[Dict[str, Any]]:
        if node.depth > self.live_depth or ctx.get("node_rank", 0) >= self.live_nodes_per_depth:
            return []
        compact_obs = [
            {
                "tool_name": r.get("tool_name"),
                "parameters": r.get("parameters"),
                "status": r.get("status"),
                "content": str(r.get("content", ""))[:900],
            }
            for r in node.observations[-8:]
        ]
        prompt = {
            "role": "next_action_proposer",
            "instruction": ctx.get("instruction", ""),
            "scenario": ctx.get("scenario"),
            "spec": ctx.get("spec"),
            "tool_history": node.tool_history[-12:],
            "tool_observations": compact_obs,
            "pinned": node.pinned,
            "db_entities": ctx.get("db_summary"),
            "tool_schema": ctx.get("tool_schema"),
            "evidence": {
                "vision_entities": (ctx.get("evidence") or {}).get("vision_entities", [])[:15],
                "canonical_matches": (ctx.get("evidence") or {}).get("canonical_matches", {}),
                "ocr_text": ((ctx.get("evidence") or {}).get("ocr_evidence") or {}).get("visible_text", [])[:30],
            },
            "output_contract": "Return JSON array of at most six next tool calls. Do not output a full chain unless each call is the immediate next alternative action. No prose.",
            "runtime_forbidden": ["ground_truth", "analysis", "final hidden metadata"],
        }
        messages = [
            {"role": "system", "content": "You are a next-action proposer for EgoBench. Propose only legal next tool calls grounded in DB/evidence/observations. No GT."},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ]
        resp = call_gpt55(messages, image_url=ctx.get("image_url", "") if node.depth == 0 else "", max_tokens=900)
        out: List[Dict[str, Any]] = []
        if not resp.get("ok"):
            return [{"source": "V32NativeVisionProposer", "action": {}, "score": -5.0, "reason": "api_error:" + str(resp.get("error", ""))[:200], "api_error": resp.get("error")}]
        ok, calls, _ = normalize_tool_calls(resp.get("text", ""))
        if not ok:
            return []
        seen = set()
        for call in calls:
            action = _valid_action(node.db, call)
            if action and _call_key(action) not in seen:
                seen.add(_call_key(action))
                out.append({"source": "V32NativeVisionProposer", "action": action, "score": 1.7, "reason": "live GPT-5.5 next-action"})
            if len(out) >= limit:
                break
        return out


def default_proposers() -> List[Any]:
    return [
        CachedTrajectoryProposer(CODEX / "analysis" / "v32_native_vision_records_v32_native_full_20260621_153428.jsonl", "V32CachedTrajectoryProposer"),
        ProgramPriorProposer(),
        DeterministicLegalActionProposer(),
        V21RetailProposer(),
        V32NativeVisionProposer(),
    ]
