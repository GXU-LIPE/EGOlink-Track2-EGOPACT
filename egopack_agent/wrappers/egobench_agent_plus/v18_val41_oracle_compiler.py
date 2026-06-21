#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V18 val41 oracle compiler.

This module is intentionally task-specific and val41-only. It exists to test
the executable compiler/resolver upper bound, not to generalize or run final.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


CODEX = Path(os.environ.get("CODEX_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/codex"))
DEFAULT_DIR = CODEX / "gt_distill_v18_val41_oracle"


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return default


def resolve_template(value: Any, slots: Dict[str, Dict[str, Any]], trace: Dict[str, Any]) -> Any:
    if isinstance(value, dict):
        if set(value.keys()) == {"literal"}:
            return copy.deepcopy(value["literal"])
        if set(value.keys()) == {"slot_ref"}:
            ref = value["slot_ref"]
            slot = slots.get(ref)
            trace.setdefault("slot_filled", []).append({"slot_ref": ref, "slot": copy.deepcopy(slot)})
            return copy.deepcopy(slot.get("value") if isinstance(slot, dict) else None)
        return {k: resolve_template(v, slots, trace) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve_template(v, slots, trace) for v in value]
    return copy.deepcopy(value)


class V18Val41OracleCompiler:
    def __init__(self, oracle_dir: Optional[str] = None) -> None:
        self.oracle_dir = Path(oracle_dir) if oracle_dir else DEFAULT_DIR
        self.tool_index = read_json(self.oracle_dir / "oracle_tool_skeleton_index.json", {})
        self.slot_index = read_json(self.oracle_dir / "oracle_slot_resolver_index.json", {})
        self.mapping = read_json(self.oracle_dir / "oracle_task_to_gt_signature.json", {})
        self.branch_index = read_json(self.oracle_dir / "oracle_branch_compiler_index.json", {})
        self.closure_index = read_json(self.oracle_dir / "oracle_closure_repair_index.json", {})
        self.manifest = read_json(self.oracle_dir / "oracle_policy_manifest.json", {})

    def lookup_pool_id(self, spec: str, subset_index: Optional[int] = None, source_original_index: Optional[int] = None, task_id: Optional[int] = None) -> Optional[str]:
        keys = []
        if subset_index is not None:
            keys.append(f"{spec}::subset_index::{int(subset_index)}")
        if source_original_index is not None:
            keys.append(f"{spec}::source_index::{int(source_original_index)}")
        if task_id is not None:
            keys.append(f"{spec}::task_id::{task_id}")
        for key in keys:
            hit = self.mapping.get(key)
            if isinstance(hit, dict) and hit.get("pool_id"):
                return hit["pool_id"]
        return None

    def compile(self, spec: str, subset_index: Optional[int] = None, source_original_index: Optional[int] = None, task_id: Optional[int] = None) -> Dict[str, Any]:
        pool_id = self.lookup_pool_id(spec, subset_index=subset_index, source_original_index=source_original_index, task_id=task_id)
        trace: Dict[str, Any] = {
            "oracle_rule_hit": bool(pool_id),
            "pool_id": pool_id,
            "lookup": {
                "spec": spec,
                "subset_index": subset_index,
                "source_original_index": source_original_index,
                "task_id": task_id,
            },
            "slot_filled": [],
            "compiled_tool_chain": [],
            "repair_applied": [],
            "final_selected_trajectory": "none",
            "uses_val41_gt_oracle_rules": True,
            "for_final_candidate": False,
        }
        if not pool_id:
            trace["error"] = "oracle_rule_not_found"
            return {"ok": False, "tool_calls": [], "trace": trace}

        tool_rule = self.tool_index.get(pool_id, {})
        slot_rule = self.slot_index.get(pool_id, {})
        slots = slot_rule.get("slots", {}) if isinstance(slot_rule, dict) else {}
        calls: List[Dict[str, Any]] = []
        for step in tool_rule.get("steps", []):
            params = resolve_template(step.get("param_template", {}), slots, trace)
            calls.append({"tool_name": step.get("tool_name", ""), "parameters": params})
        trace["compiled_tool_chain"] = copy.deepcopy(calls)

        closure = self.closure_index.get(pool_id, {})
        if isinstance(closure, dict) and closure.get("append_if_missing"):
            present = [c["tool_name"] for c in calls]
            for required in closure["append_if_missing"]:
                if required not in present:
                    trace["repair_applied"].append({"type": "missing_closure_unresolved", "tool": required})
        if not trace["repair_applied"]:
            trace["repair_applied"].append({"type": "none", "reason": "oracle_skeleton_complete"})
        trace["branch_conditions"] = (self.branch_index.get(pool_id) or {}).get("branch_conditions", [])
        trace["final_selected_trajectory"] = "oracle_compiled_skeleton"
        return {"ok": True, "tool_calls": calls, "trace": trace}


def compile_val41_oracle_action(spec: str, subset_index: Optional[int] = None, source_original_index: Optional[int] = None, task_id: Optional[int] = None, oracle_dir: Optional[str] = None) -> Dict[str, Any]:
    return V18Val41OracleCompiler(oracle_dir=oracle_dir).compile(
        spec,
        subset_index=subset_index,
        source_original_index=source_original_index,
        task_id=task_id,
    )
