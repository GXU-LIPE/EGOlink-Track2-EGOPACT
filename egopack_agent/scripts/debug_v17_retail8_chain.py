#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single-sample V17 closed-loop debug for retail8.

Read-only diagnostic:
user utterance -> visual slot -> canonical entity -> selected skeleton ->
compiled/predicted tool chain -> GT diff.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


CODEX = Path(os.environ.get("CODEX_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/codex"))
EGO = Path(os.environ.get("EGO_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/EgoBench"))
RUN_ID = os.environ.get("TRACK2_DEBUG_RUN_ID", "v17_smoke5_20260620_1140")
SPEC = "retail8"
CANDIDATE = "V17_compiler_repaired_smoke5"
MODEL = "gpt-5.5"


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return default


def read_jsonl(path: Path, limit: int = 20) -> List[Dict[str, Any]]:
    out = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
        if len(out) >= limit:
            break
    return out


def canonical(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s&']", "", text)
    return re.sub(r"\s+", " ", text).strip()


def iter_tool_calls_from_result(result_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []
    for task in result_rows or []:
        for log in task.get("tool_calls") or []:
            if isinstance(log, dict):
                for call in log.get("calls") or []:
                    if isinstance(call, dict):
                        calls.append(call)
        for log in task.get("tool_logs") or []:
            for call in log.get("calls") or []:
                if isinstance(call, dict):
                    calls.append(call)
    return calls


def iter_dialogue(result_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not result_rows:
        return []
    return result_rows[0].get("dialogue") or []


def extract_user_utterances(sample: Dict[str, Any], result_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    dialogue = iter_dialogue(result_rows)
    result_user_turns = []
    for turn in dialogue:
        if not isinstance(turn, dict):
            continue
        if "user" in turn:
            result_user_turns.append(turn.get("user"))
        elif turn.get("role") == "user":
            result_user_turns.append(turn.get("content"))
    return {
        "scenario_user_instruction": sample.get("user_instruction") or sample.get("instruction") or sample.get("Instruction") or sample.get("task") or "",
        "image_description": sample.get("image_description") or "",
        "analysis": sample.get("analysis") or sample.get("Analysis") or sample.get("task_analysis") or "",
        "simulated_user_turns": result_user_turns[:8],
    }


def extract_gt_calls(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    keys = [
        "tool_calls",
        "ground_truth_tool_calls",
        "gt_tool_calls",
        "ground_truth",
        "target_tool_calls",
        "actions",
    ]
    for key in keys:
        val = sample.get(key)
        if isinstance(val, list):
            if val and isinstance(val[0], dict):
                return val
        if isinstance(val, dict):
            for subkey in keys:
                sub = val.get(subkey)
                if isinstance(sub, list):
                    return sub
    # EgoBench scenarios often store GT in nested evaluation metadata.
    for key, val in sample.items():
        if isinstance(val, list) and key.lower().endswith(("tools", "tool_chain", "trajectory")):
            return [x for x in val if isinstance(x, dict)]
    return []


def flatten_tool_names(calls: Iterable[Dict[str, Any]]) -> List[str]:
    return [str(c.get("tool_name") or c.get("name") or "") for c in calls if isinstance(c, dict)]


def tool_sig(call: Dict[str, Any]) -> str:
    params = call.get("parameters") if isinstance(call.get("parameters"), dict) else {}
    fields = []
    for key in ("product_name", "dish_name", "set_meal_name", "ingredient_name", "recipe_name", "category", "user_id", "quantity"):
        if key in params:
            fields.append(f"{key}={canonical(params.get(key))}")
    return f"{call.get('tool_name')}({'; '.join(fields)})"


def lcs(a: List[str], b: List[str]) -> Tuple[int, List[Tuple[int, int, str]]]:
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            if a[i] == b[j]:
                dp[i][j] = dp[i + 1][j + 1] + 1
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])
    i = j = 0
    pairs = []
    while i < n and j < m:
        if a[i] == b[j]:
            pairs.append((i, j, a[i]))
            i += 1
            j += 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            i += 1
        else:
            j += 1
    return dp[0][0], pairs


def classify_intent(goal: str) -> str:
    text = (goal or "").lower()
    if any(x in text for x in ["if ", "otherwise", "else", "whether", "exceeds", "less than", "greater than"]):
        return "branch_then_mutation"
    if any(x in text for x in ["highest", "lowest", "cheapest", "most", "least"]):
        return "ranking_filtering"
    if any(x in text for x in ["add", "remove", "cart", "order", "menu", "shopping list"]):
        return "cart_order_mutation"
    if any(x in text for x in ["total", "tax", "payment", "nutrition", "summary"]):
        return "aggregate_required"
    if any(x in text for x in ["point", "visible", "image", "video", "shelf", "left", "right", "front", "behind", "next to"]):
        return "visual_query"
    return "query_only"


def load_selected_skeleton(goal: str) -> Dict[str, Any]:
    intent = classify_intent(goal)
    skel = read_json(CODEX / "gt_distill_v17" / "tool_skeleton_index.json", {})
    slots = read_json(CODEX / "gt_distill_v17" / "slot_resolver_index.json", {})
    branch = read_json(CODEX / "gt_distill_v17" / "branch_compiler_index.json", {})
    closure = read_json(CODEX / "gt_distill_v17" / "closure_repair_index.json", {})
    anti = read_json(CODEX / "gt_distill_v17" / "anti_broad_scan_index.json", {})
    keys = [f"retail::{intent}", "retail::branch_then_mutation", "retail::cart_order_mutation", "retail::aggregate_required"]
    def pick(bank: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        index = bank.get("index") or {}
        for key in keys:
            out.extend(index.get(key) or [])
        return out[:limit]
    return {
        "intent": intent,
        "candidate_keys": keys,
        "skeleton_cards": pick(skel, 3),
        "slot_cards": pick(slots, 3),
        "branch_cards": pick(branch, 2),
        "closure_cards": pick(closure, 2),
        "anti_cards": ((anti.get("index") or {}).get("retail") or [])[:8],
    }


def find_visual_cards() -> Dict[str, Any]:
    candidates = []
    roots = [
        CODEX / "visual_cache_v12" / "qwen3vl_grounding",
        CODEX / "visual_cache_v12" / "qwen3vl_grounding_by_video",
        CODEX / "visual_cache",
    ]
    patterns = ["*retail8*.json", "*retail_8*.json", "*retail8*.txt", "*retail_8*.txt"]
    for root in roots:
        if not root.exists():
            continue
        for pat in patterns:
            candidates.extend(sorted(root.glob(pat))[:20])
    cards = []
    for p in candidates[:30]:
        if p.suffix.lower() == ".json":
            data = read_json(p, {})
        else:
            data = {"text": p.read_text(encoding="utf-8", errors="replace")[:4000]}
        cards.append({"path": str(p), "data": data})
    return {"candidate_count": len(candidates), "cards": cards[:10]}


def extract_visual_slot(visual: Dict[str, Any], utterances: Dict[str, Any]) -> Dict[str, Any]:
    text_parts = [
        utterances.get("scenario_user_instruction", ""),
        utterances.get("image_description", ""),
        utterances.get("analysis", ""),
    ]
    for card in visual.get("cards") or []:
        data = card.get("data")
        text_parts.append(json.dumps(data, ensure_ascii=False)[:6000])
    text = "\n".join(str(x or "") for x in text_parts)
    lower = text.lower()
    product_candidates = []
    top_k_candidates = []

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                lk = str(k).lower()
                if lk in {"top_k_candidates", "top_candidates", "product_candidates", "candidate_products"} and isinstance(v, list):
                    for item in v:
                        if isinstance(item, str):
                            top_k_candidates.append(item)
                        elif isinstance(item, dict):
                            for name_key in (
                                "product_name", "canonical_name", "entity_name", "name",
                                "candidate", "label", "text", "value",
                            ):
                                if item.get(name_key):
                                    top_k_candidates.append(str(item.get(name_key)))
                                    break
                if lk in {"product_name", "canonical_name", "entity_name"} and isinstance(v, str):
                    product_candidates.append(v)
                walk(v)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    for card in visual.get("cards") or []:
        walk(card.get("data"))
    for m in re.finditer(r'"product_name"\s*:\s*"([^"]+)"', text):
        product_candidates.append(m.group(1))
    for m in re.finditer(r'"name"\s*:\s*"([^"]+)"', text):
        val = m.group(1)
        if any(tok in val.lower() for tok in ["crispy", "grisbi", "choco", "cookie", "prosecco", "beer", "wine"]):
            product_candidates.append(val)
    # Conservative textual cues only; no hidden GT use.
    relation = []
    for cue in ["left", "right", "front", "behind", "next to", "adjacent", "pointed", "upper", "lower", "shelf"]:
        if cue in lower:
            relation.append(cue)
    dedup = []
    seen = set()
    for p in list(top_k_candidates) + product_candidates:
        c = canonical(p)
        if c and c not in seen:
            seen.add(c)
            dedup.append(p)
    return {
        "visual_relation_cues": relation[:20],
        "top_k_candidates_extracted": top_k_candidates[:30],
        "product_candidates_from_visual_or_context": dedup[:20],
        "has_top_k_candidates": "top_k_candidates" in lower,
        "raw_context_chars": len(text),
    }


def compare(gt_calls: List[Dict[str, Any]], pred_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
    gt_names = flatten_tool_names(gt_calls)
    pred_names = flatten_tool_names(pred_calls)
    lcs_len, pairs = lcs(gt_names, pred_names)
    gt_sigs = [tool_sig(c) for c in gt_calls]
    pred_sigs = [tool_sig(c) for c in pred_calls]
    return {
        "gt_tool_names": gt_names,
        "pred_tool_names": pred_names,
        "gt_signatures": gt_sigs,
        "pred_signatures": pred_sigs,
        "lcs_tool_name_matches": lcs_len,
        "lcs_pairs": pairs,
        "missing_gt_tools_by_name": [x for x in gt_names if x not in pred_names],
        "extra_pred_tools_by_name": [x for x in pred_names if x not in gt_names],
    }


def render_md(report: Dict[str, Any]) -> str:
    lines = [
        f"# V17 Retail8 Closed Loop Debug {report['generated_at']}",
        "",
        "- sample: retail8 first smoke5 item",
        "- run_id: " + RUN_ID,
        "- candidate: " + CANDIDATE,
        "- final_run: false",
        "- uses_final_hidden_metadata: false",
        "- uses_val41_gt_for_policy: false",
        "- gt_used_for_posthoc_diff_only: true",
        "",
        "## 1. User Utterance",
        "",
        "Scenario user instruction:",
        "```text",
        str(report["utterances"].get("scenario_user_instruction") or "")[:3000],
        "```",
        "",
        "Simulated user turns:",
        "```text",
        "\n".join(str(x) for x in report["utterances"].get("simulated_user_turns") or [])[:3000],
        "```",
        "",
        "## 2. Visual Slot",
        "",
        "```json",
        json.dumps(report["visual_slot"], ensure_ascii=False, indent=2)[:6000],
        "```",
        "",
        "## 3. Canonical Entity",
        "",
        "```json",
        json.dumps(report["canonical_entity"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## 4. Selected Skeleton",
        "",
        f"- intent: `{report['selected_skeleton'].get('intent')}`",
        f"- candidate_keys: `{report['selected_skeleton'].get('candidate_keys')}`",
        "",
        "Skeleton cards:",
        "```json",
        json.dumps(report["selected_skeleton"].get("skeleton_cards"), ensure_ascii=False, indent=2)[:7000],
        "```",
        "",
        "## 5. Compiled Tool Chain",
        "",
        "Predicted/executed V17 repaired chain:",
        "```json",
        json.dumps(report["compiled_tool_chain"], ensure_ascii=False, indent=2)[:10000],
        "```",
        "",
        "## 6. GT Diff",
        "",
        "```json",
        json.dumps(report["gt_diff"], ensure_ascii=False, indent=2)[:10000],
        "```",
        "",
        "## Breakpoint Diagnosis",
        "",
    ]
    for item in report["breakpoints"]:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def main() -> None:
    smoke_dir = CODEX / "state" / "materialized_splits" / f"v17_smoke5_{RUN_ID}"
    sample_path = smoke_dir / f"{SPEC}.json"
    result_path = EGO / "results" / f"{MODEL}-{CANDIDATE}-{RUN_ID}" / f"{SPEC}_easy.json"
    wrapper_dir = CODEX / "runs" / "V17_GT100_EXECUTABLE_COMPILER_SMOKE5" / RUN_ID / "wrapper_events"
    sample_rows = read_json(sample_path, [])
    result_rows = read_json(result_path, [])
    if not sample_rows:
        raise SystemExit(f"missing sample rows: {sample_path}")
    sample = sample_rows[0]
    utterances = extract_user_utterances(sample, result_rows)
    visual = find_visual_cards()
    visual_slot = extract_visual_slot(visual, utterances)
    pred_calls = iter_tool_calls_from_result(result_rows)
    gt_calls = extract_gt_calls(sample)
    goal = utterances.get("scenario_user_instruction") or "\n".join(str(x) for x in utterances.get("simulated_user_turns") or [])
    selected = load_selected_skeleton(goal)
    candidates = visual_slot.get("product_candidates_from_visual_or_context") or []
    canonical_entity = {
        "selected_candidate_by_current_debugger": candidates[0] if candidates else "",
        "canonicalized": canonical(candidates[0]) if candidates else "",
        "all_candidates": candidates[:20],
        "note": "This is diagnostic extraction from existing visual/context cards, not a model call.",
    }
    diff = compare(gt_calls, pred_calls)
    events = []
    for p in sorted(wrapper_dir.glob("*.jsonl")):
        # Smoke files are single-task per spec, so task_id is 1 for each materialized spec.
        events.extend(read_jsonl(p, limit=200))
    v17_events = [e for e in events if e.get("event") == "v17_compiler_repair"]
    breakpoints = []
    if not utterances.get("scenario_user_instruction") and not utterances.get("simulated_user_turns"):
        breakpoints.append("FAIL user utterance: no usable user goal was extracted.")
    else:
        breakpoints.append("OK user utterance extracted.")
    if not visual_slot.get("product_candidates_from_visual_or_context"):
        breakpoints.append("FAIL visual slot: no product candidates surfaced from visual/context cards.")
    elif not visual_slot.get("has_top_k_candidates"):
        breakpoints.append("WEAK visual slot: candidates exist, but no explicit top_k_candidates field was found.")
    else:
        breakpoints.append("OK visual slot has top_k_candidates and product candidates.")
    if not canonical_entity.get("canonicalized"):
        breakpoints.append("FAIL canonical entity: no candidate was available to canonicalize.")
    else:
        breakpoints.append("OK canonical entity string exists, but DB membership is not proven by this debugger.")
    if not selected.get("skeleton_cards"):
        breakpoints.append("FAIL selected skeleton: V17 index returned no skeleton cards for inferred intent.")
    else:
        breakpoints.append("OK selected skeleton cards found.")
    if not pred_calls:
        breakpoints.append("FAIL compiled tool chain: no executed tool calls in result.")
    else:
        breakpoints.append(f"OK compiled/executed tool chain has {len(pred_calls)} calls.")
    if diff.get("lcs_tool_name_matches", 0) == 0:
        breakpoints.append("FAIL GT diff: predicted tool names have zero LCS overlap with GT tool names.")
    else:
        breakpoints.append(f"GT diff: {diff.get('lcs_tool_name_matches')} tool-name LCS matches.")
    if not v17_events:
        breakpoints.append("FAIL/WEAK V17 repair: no v17_compiler_repair event was observed for this sample.")
    else:
        breakpoints.append(f"OK V17 repair events observed: {len(v17_events)}.")

    report = {
        "generated_at": time.strftime("%Y%m%d_%H%M%S"),
        "paths": {
            "sample_path": str(sample_path),
            "result_path": str(result_path),
            "wrapper_events_dir": str(wrapper_dir),
        },
        "utterances": utterances,
        "visual_card_audit": visual,
        "visual_slot": visual_slot,
        "canonical_entity": canonical_entity,
        "selected_skeleton": selected,
        "compiled_tool_chain": {
            "pred_tool_names": flatten_tool_names(pred_calls),
            "pred_calls": pred_calls,
            "v17_repair_events": v17_events,
        },
        "gt_calls": gt_calls,
        "gt_diff": diff,
        "breakpoints": breakpoints,
    }
    out_json = CODEX / "analysis" / f"V17_RETAIL8_CHAIN_DEBUG_{report['generated_at']}.json"
    out_md = CODEX / "reports" / f"V17_RETAIL8_CHAIN_DEBUG_{report['generated_at']}.md"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(render_md(report), encoding="utf-8")
    print(json.dumps({"report": str(out_md), "json": str(out_json), "breakpoints": breakpoints}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
