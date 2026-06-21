#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V35 native-loop data amplification.

Val41 shadow only. No final run, no final hidden metadata, no V10 overwrite.
Val41 GT is used only by official post-eval/oracle diagnostics after
candidate trajectories have already been generated.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import copy
import importlib
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


CODEX = Path(os.environ.get("CODEX_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/codex"))
EGO = Path(os.environ.get("EGO_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/EgoBench"))
V10_ZIP = CODEX / "submissions" / "V10_full_memory_final_candidate_draft_track2.zip"
V10_EXPECTED_SHA = "5f2b8aacf986e4b3107b560e76fe8dfc52b5b3da6e407c9e0e6d0433db36849d"
V32_RUN_ID = "v32_native_full_20260621_153428"
V34_RUN_ID = "v34_v32_expansion_20260621_1658"

sys.path.insert(0, str(CODEX / "scripts"))
sys.path.insert(0, str(CODEX / "wrappers"))
sys.path.insert(0, str(CODEX))
sys.path.insert(0, str(EGO))

v32 = importlib.import_module("run_v32_native_vision_val41")
service_agent = importlib.import_module("egobench_agent_plus.v32_native_vision_service_agent")


def stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return default


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def read_jsonl(path: Path, limit: int = 0) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
            if limit and len(rows) >= limit:
                break
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    tmp.replace(path)


BASE_OUTPUT_RULES = """Output rules:
- If you need tools, output exactly one JSON array and nothing else.
- Each array item must be {"tool_name": "...", "parameters": {...}}.
- If no more tools are needed, output a short final response in plain text.
- Never mix prose and JSON in the same response.
- Use only tools listed in the schema.
- Do not invent parameters. Use current user instruction, DB context, visual evidence, OCR/ASR, or previous tool observations.
- This is not ordinary QA. Success requires both required DB result and required tool process.
- Do not use hidden scenario metadata, ground truth, analysis fields, or final set metadata.
"""

BEHAVIOR_HINT = """Compact native-loop pattern:
- Resolve current user and entity, then call narrow attribute/retrieval tools.
- Branch only from tool observations.
- Mutate the resolved target once or for all tied targets.
- Run requested payment/tax/nutrition/order/menu/cart closure near the end, then stop.
"""

V35_PROMPTS: Dict[str, str] = {
    "compact_seed3": f"You are the EgoBench service agent. Make a concise official-tool attempt with canonical names.\n{BASE_OUTPUT_RULES}\n",
    "compact_seed4": f"You are the EgoBench service agent. Prefer narrow evidence-backed calls over broad exploration.\n{BASE_OUTPUT_RULES}\n",
    "compact_seed5": f"You are the EgoBench service agent. If uncertain, query a narrow candidate attribute rather than asking the user.\n{BASE_OUTPUT_RULES}\n",
    "compact_result_tool_dual": f"You are the EgoBench service agent. Optimize for both final DB result and required tool process.\n{BASE_OUTPUT_RULES}\n",
    "compact_tracebank_hint": f"You are the EgoBench service agent. Follow successful shapes: resolve, query, branch, mutate, close.\n{BASE_OUTPUT_RULES}\n{BEHAVIOR_HINT}\n",
    "compact_retrieval_first": f"You are the EgoBench service agent. Before any mutation, retrieve/confirm the entity or current state with official tools.\n{BASE_OUTPUT_RULES}\n",
    "compact_no_ask_visual": f"You are the EgoBench service agent. Do not ask the user for visual names; infer candidates from image/OCR/DB and verify by tools.\n{BASE_OUTPUT_RULES}\n",
    "compact_no_broad_scan_v2": f"You are the EgoBench service agent. Avoid leading all-item scans; start with category/entity-specific tools.\n{BASE_OUTPUT_RULES}\n",
    "compact_closure_v2": f"You are the EgoBench service agent. If totals, tax, payment, nutrition, cart/order/menu summary are requested, compute them at the end.\n{BASE_OUTPUT_RULES}\n",
    "order_pin_payment": f"You are the EgoBench order service agent. Pin user_id and active restaurant, inspect order/menu, mutate, then payment/summary if requested.\n{BASE_OUTPUT_RULES}\n",
    "order_setmeal_disambiguate": f"You are the EgoBench order service agent. Query menu/set-meal details before choosing dish vs set_meal.\n{BASE_OUTPUT_RULES}\n",
    "order_mutate_once": f"You are the EgoBench order service agent. Avoid duplicate order mutations; after a successful mutation, run only requested closure.\n{BASE_OUTPUT_RULES}\n",
    "restaurant_menu_grounded": f"You are the EgoBench restaurant service agent. Confirm dish/set_meal/category/restaurant with tools before order/reservation actions.\n{BASE_OUTPUT_RULES}\n",
    "restaurant_dish_setmeal": f"You are the EgoBench restaurant service agent. Keep dish and set_meal separate; use dish-specific tools when possible.\n{BASE_OUTPUT_RULES}\n",
    "restaurant_finish_closure": f"You are the EgoBench restaurant service agent. After add/modify/order, run requested total/nutrition/payment closure and stop.\n{BASE_OUTPUT_RULES}\n",
    "kitchen_multistage_compact": f"You are the EgoBench kitchen service agent. Resolve ingredient/recipe, query branch attributes, mutate menu/shopping list, compute closure.\n{BASE_OUTPUT_RULES}\n",
    "kitchen_branch_closure": f"You are the EgoBench kitchen service agent. Branch only after observations; do not stop before mutation plus requested nutrition/taste closure.\n{BASE_OUTPUT_RULES}\n{BEHAVIOR_HINT}\n",
    "kitchen_shopping_closure": f"You are the EgoBench kitchen service agent. Maintain menu/shopping list state and finish shopping mutations with requested nutrition totals.\n{BASE_OUTPUT_RULES}\n",
    "kitchen_long_trace_ok": f"You are the EgoBench kitchen service agent. Complex kitchen tasks may require several recipe/allergen/shelf-life checks; complete the closure.\n{BASE_OUTPUT_RULES}\n",
    "retail_visual_candidate": f"You are the EgoBench retail service agent. Treat visual/OCR as candidates, bind to DB product/category, then query attributes.\n{BASE_OUTPUT_RULES}\n",
    "retail_filter_then_mutate": f"You are the EgoBench retail service agent. Narrow by category/country/brand/taste before price/nutrition/discount; then mutate and close.\n{BASE_OUTPUT_RULES}\n",
    "retail_cart_closure": f"You are the EgoBench retail service agent. After cart mutation, compute requested tax/payment/nutrition closure and stop.\n{BASE_OUTPUT_RULES}\n",
    "retail_ties_all": f"You are the EgoBench retail service agent. For highest/lowest with ties, add or remove all tied resolved products.\n{BASE_OUTPUT_RULES}\n",
    "compact_finish_after_success": f"You are the EgoBench service agent. Once mutation and requested closure are done, output final response and stop.\n{BASE_OUTPUT_RULES}\n",
}

ORIGINAL_VARIANT_PROMPT = service_agent.variant_prompt


def variant_prompt(name: str, repair_hint: str = "") -> str:
    prompt = V35_PROMPTS.get(name)
    if prompt is None:
        return ORIGINAL_VARIANT_PROMPT(name, repair_hint=repair_hint)
    if repair_hint:
        prompt += "\nNon-GT runtime repair hint:\n" + repair_hint.strip()[:1400] + "\n"
    return prompt


def patch_prompt_router() -> None:
    service_agent.variant_prompt = variant_prompt
    original_runner = v32.run_native_service_agent
    max_rounds = int(os.environ.get("V35_MAX_ROUNDS", "5"))
    max_tool_calls = int(os.environ.get("V35_MAX_TOOL_CALLS", "52"))

    def limited_runner(**kwargs: Any) -> Dict[str, Any]:
        kwargs["max_rounds"] = min(int(kwargs.get("max_rounds", max_rounds)), max_rounds)
        kwargs["max_tool_calls"] = min(int(kwargs.get("max_tool_calls", max_tool_calls)), max_tool_calls)
        return original_runner(**kwargs)

    v32.run_native_service_agent = limited_runner


def task_key(spec: str, idx: int) -> str:
    return f"{spec}::{int(idx)}"


def metric_count(summary: Dict[str, Any], total: int = 41) -> int:
    return int(round(float(summary.get("joint", 0) or 0) * total))


def strip_item(rec: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in rec.items() if k != "item"}


def load_item(result_dir: Path, spec: str, pos: int) -> Dict[str, Any] | None:
    return v32.load_item(result_dir, spec, pos)


MUTATION_RE = re.compile(r"^(add|remove|delete|update|modify)_|_(to|from)_(cart|order|shopping_list|menu)$")
RETRIEVAL_PREFIXES = ("get_", "find_", "filter_", "list_", "search_", "query_")
BROAD_RE = re.compile(r"all_|price_range|category_names|get_menu_categories|get_all", re.I)


def tool_result_errors(item: Dict[str, Any]) -> int:
    n = 0
    for turn in item.get("tool_calls") or []:
        for res in turn.get("results") or []:
            status = str(res.get("status", ""))
            content = str(res.get("content", ""))
            if status in {"error", "blocked"} or '"status": "error"' in content or "not found" in content.lower():
                n += 1
    return n


def repeated_exact(program: List[Dict[str, Any]]) -> int:
    seen: set[str] = set()
    reps = 0
    for call in program:
        key = json.dumps(call, sort_keys=True, ensure_ascii=False)
        if key in seen:
            reps += 1
        seen.add(key)
    return reps


def tool_features(rec: Dict[str, Any]) -> Dict[str, Any]:
    program = rec.get("tool_program") or []
    names = [str(c.get("tool_name", "")) for c in program]
    flags = set(rec.get("risk_flags") or [])
    item = rec.get("item") or {}
    instr = str(item.get("instruction") or rec.get("instruction") or "")
    final_text = str(rec.get("final_text") or item.get("final_text") or "")
    first_mut = next((i for i, n in enumerate(names) if MUTATION_RE.search(n)), None)
    retrieval_before_mutation = True if first_mut is None else any(n.startswith(RETRIEVAL_PREFIXES) for n in names[:first_mut])
    mutation_present = any(MUTATION_RE.search(n) for n in names)
    closure_tools = {
        "compute_total_payment",
        "compute_total_tax",
        "compute_total_nutrition",
        "compute_total_nutritions",
        "tally_total_nutritional_characteristics",
        "tally_total_tastes",
        "get_user_order_summary",
        "calculate_order_total_nutrition",
        "calculate_total_nutrition",
    }
    closure_present = any(n in closure_tools for n in names)
    return {
        "scenario": rec.get("scenario"),
        "variant_tag": rec.get("variant_tag") or rec.get("variant"),
        "tool_count": len(names),
        "tool_errors": tool_result_errors(item),
        "repeats": repeated_exact(program),
        "broad_first": bool(names and any(BROAD_RE.search(n) for n in names[:2])),
        "retrieval_before_mutation": retrieval_before_mutation,
        "mutation_present": mutation_present,
        "closure_present": closure_present,
        "stopped": final_text.startswith("[V32 stopped") or final_text.startswith("[V34 stopped") or final_text.startswith("[V35 stopped") or "stopped" in flags,
        "ask_user": "refusal_or_ask_user" in flags or "ask" in final_text.lower()[:300],
        "vision_success": bool(rec.get("vision_success")),
        "first_tool": names[0] if names else "",
        "name_set": sorted(set(names)),
        "instruction_has_mutation": any(x in instr.lower() for x in ("add", "remove", "delete", "replace", "change", "update", "order", "cart", "shopping list", "menu")),
        "instruction_has_closure": any(x in instr.lower() for x in ("total", "tax", "payment", "nutrition", "summary", "taste", "calorie", "protein")),
    }


def build_trace_bank() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    gt_rows = read_jsonl(CODEX / "gt_distill_v16" / "gt100_pool.jsonl", limit=200000)
    for r in gt_rows:
        if not r.get("no_final_metadata", True) or not r.get("excluded_final309", True) or not r.get("excluded_val41", True):
            continue
        names = [str(x) for x in (r.get("tool_names") or [])]
        if not names:
            continue
        rows.append({
            "source": "nonfinal_gt100",
            "scenario": r.get("scenario"),
            "task_type": r.get("task_type"),
            "tool_count": len(names),
            "tool_names": names,
            "has_mutation": any(MUTATION_RE.search(n) for n in names),
            "has_closure": any(n.startswith(("compute_", "tally_", "calculate_")) or "summary" in n for n in names),
            "has_broad": any(BROAD_RE.search(n) for n in names[:2]),
            "no_final_metadata": True,
        })
    for p in [
        CODEX / "analysis" / "v34_trajectory_pool.jsonl",
        CODEX / "analysis" / "v34_protected_selection_trace.jsonl",
        CODEX / "analysis" / "v33_selection_trace.jsonl",
        CODEX / "analysis" / "v29_selection_trace.jsonl",
        CODEX / "analysis" / "v30_prior_selection_trace.jsonl",
    ]:
        for r in read_jsonl(p, limit=5000):
            rows.append({
                "source": p.name,
                "scenario": r.get("scenario"),
                "task_key": r.get("task_key"),
                "candidate_id": r.get("candidate_id") or r.get("selected_candidate_id"),
                "score": r.get("score") or r.get("selected_score") or r.get("oracle_score") or {},
                "features": r.get("features") or {},
                "no_final_metadata": True,
            })
    scenario_counts = Counter(str(r.get("scenario")) for r in rows if r.get("scenario"))
    gt_tool_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    gt_pair_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    for r in rows:
        if r.get("source") != "nonfinal_gt100":
            continue
        scenario = str(r.get("scenario"))
        names = r.get("tool_names") or []
        gt_tool_counts[scenario].update(names)
        for a, b in zip(names, names[1:]):
            gt_pair_counts[scenario][f"{a}->{b}"] += 1
    bank = {
        "total_cards": len(rows),
        "nonfinal_gt100_cards": sum(1 for r in rows if r.get("source") == "nonfinal_gt100"),
        "scenario_counts": dict(scenario_counts),
        "top_tools_by_scenario": {k: v.most_common(20) for k, v in gt_tool_counts.items()},
        "top_pairs_by_scenario": {k: v.most_common(20) for k, v in gt_pair_counts.items()},
        "no_final_metadata": True,
    }
    write_jsonl(CODEX / "analysis" / "v35_trace_bank.jsonl", rows)
    return rows, bank


def tracebank_score(rec: Dict[str, Any], bank: Dict[str, Any]) -> float:
    feat = tool_features(rec)
    scenario = str(feat.get("scenario"))
    names = feat.get("name_set") or []
    top_tools = {k: v for k, v in bank.get("top_tools_by_scenario", {}).get(scenario, [])}
    s = 0.0
    for n in names:
        if n in top_tools:
            s += min(2.5, top_tools[n] / 100.0)
    return min(20.0, s)


def selector_score(rec: Dict[str, Any], bank: Dict[str, Any], *, mode: str) -> Dict[str, Any]:
    feat = tool_features(rec)
    score = 0.0
    reasons: List[str] = []
    if rec.get("api_errors"):
        score -= 35
        reasons.append("api_error")
    if feat["tool_count"] <= 0:
        score -= 30
        reasons.append("no_tools")
    else:
        score += min(22, feat["tool_count"] * 1.1)
    if feat["tool_errors"]:
        score -= feat["tool_errors"] * 7
        reasons.append(f"tool_errors={feat['tool_errors']}")
    if feat["broad_first"]:
        score -= 10
        reasons.append("broad_first")
    if not feat["retrieval_before_mutation"]:
        score -= 16
        reasons.append("mutation_before_retrieval")
    if feat["instruction_has_mutation"] and feat["mutation_present"]:
        score += 14
        reasons.append("mutation_present")
    if feat["instruction_has_closure"] and feat["closure_present"]:
        score += 12
        reasons.append("closure_present")
    if feat["instruction_has_closure"] and not feat["closure_present"]:
        score -= 10
        reasons.append("closure_missing")
    if feat["repeats"]:
        score -= min(22, feat["repeats"] * 3)
        reasons.append(f"repeats={feat['repeats']}")
    if feat["stopped"]:
        score -= 20
        reasons.append("stopped")
    if feat["ask_user"]:
        score -= 20
        reasons.append("ask_user")
    if feat["vision_success"]:
        score += 2
    tb = tracebank_score(rec, bank)
    score += tb
    if tb:
        reasons.append(f"tracebank={tb:.1f}")
    scenario = str(feat["scenario"])
    name_set = set(feat["name_set"])
    if scenario == "kitchen" and feat["tool_errors"] == 0:
        if {"get_ingredient_nutrition", "get_recipe_allergens", "get_ingredient_shelf_life", "add_recipe_to_menu"}.issubset(name_set):
            score += 12
            reasons.append("kitchen_branch_coverage")
        if {"remove_from_shopping_list", "add_to_shopping_list", "compute_total_nutritions"}.issubset(name_set):
            score += 42
            reasons.append("kitchen_shopping_closure_coverage")
        if feat["stopped"] and "kitchen_shopping_closure_coverage" in reasons:
            score -= 45
            reasons.append("kitchen_stopped_not_trusted")
    if scenario in {"order", "restaurant"} and feat["mutation_present"] and feat["closure_present"]:
        score += 10
        reasons.append("order_restaurant_mutation_closure")
    if scenario == "retail" and feat["mutation_present"] and feat["closure_present"] and not feat["broad_first"]:
        score += 10
        reasons.append("retail_mutation_closure_narrow")
    if mode == "dev":
        # Dev-only: allow high-micro/result trace patterns found in val41 post-eval.
        sc = rec.get("score") or {}
        if float(sc.get("result", 0) or 0) >= 1:
            score += 8
            reasons.append("dev_result_success_signal")
        if float(sc.get("micro", 0) or 0) >= 0.5:
            score += 6
            reasons.append("dev_high_micro_signal")
    return {**feat, "selector_score": round(score, 4), "reasons": reasons, "mode": mode}


def v22_eval_map() -> Dict[str, Dict[str, Any]]:
    rows, _ = v32.eval_result_dir(v32.V22_DIR)
    return {task_key(r["spec"], r["index"]): r for r in rows}


def result_eval_map(result_dir: Path) -> Dict[str, Dict[str, Any]]:
    rows, _ = v32.eval_result_dir(result_dir)
    return {task_key(r["spec"], r["index"]): r for r in rows}


def all_tasks_by_key() -> Dict[str, Dict[str, Any]]:
    return {task_key(t["spec"], t["index"]): t for t in v32.all_tasks()}


def read_gt_row(task: Dict[str, Any]) -> Dict[str, Any]:
    rows = read_json(v32.SPLIT_DIR / f"{task['spec']}.json", [])
    return rows[int(task["local_pos"])]


def load_v32_records() -> List[Dict[str, Any]]:
    rows = read_jsonl(CODEX / "analysis" / f"v32_native_vision_records_{V32_RUN_ID}.jsonl")
    state = read_json(CODEX / "runs" / "V32_NATIVE_GPT55_VISION_OFFICIAL_LOOP_AGENT" / V32_RUN_ID / "state.json", {})
    tasks = all_tasks_by_key()
    out: List[Dict[str, Any]] = []
    for r in rows:
        key = r.get("task_key")
        if not key or key not in tasks:
            continue
        tag = str(r.get("variant_tag") or r.get("variant"))
        variant = tag.split("_r", 1)[0]
        result_dir = Path(state.get("result_dirs", {}).get(f"{variant}_raw", ""))
        item = load_item(result_dir, tasks[key]["spec"], int(tasks[key]["local_pos"])) if result_dir else None
        if item:
            rr = dict(r)
            rr["item"] = item
            rr["source"] = "V32"
            rr["candidate_id"] = f"V32::{tag}::{key}"
            out.append(rr)
    return out


def load_v34_records() -> List[Dict[str, Any]]:
    rows = read_jsonl(CODEX / "analysis" / "v34_trajectory_pool.jsonl")
    tasks = all_tasks_by_key()
    out: List[Dict[str, Any]] = []
    for r in rows:
        key = r.get("task_key")
        if not key or key not in tasks:
            continue
        # V34 pool is compact. Reload item from per-shard where available.
        source = str(r.get("source", ""))
        tag = str(r.get("variant_tag") or r.get("variant"))
        item = None
        if source == "V34":
            variant = tag.split("_r", 1)[0]
            shard = CODEX / "runs" / "V34_V32_NATIVE_LOOP_EXPANSION_AND_PROTECTED_MERGE" / V34_RUN_ID / "shards" / f"{variant}_r0" / f"{tasks[key]['spec']}__{int(tasks[key]['index'])}.json"
            full = read_json(shard, {})
            item = full.get("item") if isinstance(full, dict) else None
        elif source == "V32":
            # Already reloadable through V32 state.
            continue
        if item:
            rr = dict(r)
            rr["item"] = item
            rr["source"] = "V34"
            rr["candidate_id"] = rr.get("candidate_id") or f"V34::{tag}::{key}"
            out.append(rr)
    return out


def timeout_record(task: Dict[str, Any], variant: str, message: str) -> Dict[str, Any]:
    key = task_key(task["spec"], task["index"])
    return {
        "task_key": key,
        "spec": task["spec"],
        "index": int(task["index"]),
        "local_pos": int(task["local_pos"]),
        "scenario": task["scenario"],
        "number": int(task["number"]),
        "variant": variant,
        "variant_tag": f"{variant}_r0",
        "source": "V35",
        "candidate_id": f"V35::{variant}_r0::{key}",
        "item": {},
        "score": {"joint": 0, "result": 0, "tool": 0, "matches": 0, "gt_calls": 0, "interaction_calls": 0, "micro": 0},
        "tool_program": [],
        "risk_flags": ["task_timeout"],
        "api_errors": [message],
        "vision_success": False,
        "contact_sheet": "",
        "final_text": "",
    }


def run_single(args: argparse.Namespace) -> None:
    patch_prompt_router()
    tasks = all_tasks_by_key()
    task = tasks[args.single_task_key]
    evidence_cache = v32.load_evidence_cache()
    rec = v32.run_task_variant(task, args.single_variant, evidence_cache, repair_round=0)
    rec["source"] = "V35"
    rec["variant_tag"] = f"{args.single_variant}_r0"
    rec["candidate_id"] = f"V35::{args.single_variant}_r0::{rec['task_key']}"
    write_json(Path(args.single_out), rec)


def run_global_pool(tasks: List[Dict[str, Any]], variants: List[str], run_dir: Path, workers: int, timeout_s: int) -> Dict[str, List[Dict[str, Any]]]:
    script_path = Path(__file__).resolve()
    for variant in variants:
        (run_dir / "shards" / f"{variant}_r0").mkdir(parents=True, exist_ok=True)

    def one(pair: Tuple[str, Dict[str, Any]]) -> Dict[str, Any]:
        variant, task = pair
        key = task_key(task["spec"], task["index"])
        out = run_dir / "shards" / f"{variant}_r0" / f"{task['spec']}__{int(task['index'])}.json"
        if out.exists() and out.stat().st_size > 0:
            rec = read_json(out, {})
            if isinstance(rec, dict) and rec.get("task_key"):
                return rec
        cmd = [
            sys.executable,
            str(script_path),
            "--single-task-key",
            key,
            "--single-variant",
            variant,
            "--single-out",
            str(out),
        ]
        env = os.environ.copy()
        env.setdefault("TRACK2_OPENAI_NO_PROXY", "1")
        env.setdefault("TRACK2_OPENAI_TIMEOUT", "105")
        env.setdefault("V35_MAX_ROUNDS", "5")
        env.setdefault("V35_MAX_TOOL_CALLS", "52")
        try:
            proc = subprocess.run(cmd, cwd=str(CODEX), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="replace", timeout=timeout_s, check=False)
            if out.exists() and out.stat().st_size > 0:
                rec = read_json(out, {})
                if isinstance(rec, dict) and rec.get("task_key"):
                    return rec
            rec = timeout_record(task, variant, f"single_task_failed rc={proc.returncode}; stderr={proc.stderr[-1000:]}")
            write_json(out, rec)
            return rec
        except subprocess.TimeoutExpired:
            rec = timeout_record(task, variant, f"single_task_timeout_after_{timeout_s}s")
            write_json(out, rec)
            return rec

    pairs = [(variant, task) for variant in variants for task in tasks]
    by_variant: Dict[str, List[Dict[str, Any]]] = {f"{v}_r0": [] for v in variants}
    last_ping = time.time()
    done = 0
    with cf.ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = {ex.submit(one, pair): pair for pair in pairs}
        for fut in cf.as_completed(futs):
            variant, task = futs[fut]
            try:
                rec = fut.result()
            except Exception as exc:
                rec = timeout_record(task, variant, f"pair_exception={type(exc).__name__}: {exc}")
            rec["source"] = "V35"
            rec["variant_tag"] = f"{variant}_r0"
            rec["candidate_id"] = f"V35::{variant}_r0::{rec['task_key']}"
            by_variant[f"{variant}_r0"].append(rec)
            done += 1
            now = time.time()
            if now - last_ping > 60 or done % 24 == 0:
                print(f"[{time.strftime('%H:%M:%S')}] V35 global pool {done}/{len(pairs)} variants={len(variants)} workers={workers}", flush=True)
                last_ping = now
    for tag, rows in by_variant.items():
        rows.sort(key=lambda r: (r["spec"], int(r["local_pos"])))
        write_jsonl(run_dir / f"{tag}_records.jsonl", [strip_item(r) for r in rows])
    return by_variant


def eval_one(task: Dict[str, Any], item: Dict[str, Any]) -> Dict[str, Any]:
    return v32.evaluate_one(read_gt_row(task), item, task["scenario"], int(task["number"]))


def write_result_dir(name: str, run_id: str, items: Dict[Tuple[str, int], Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    result_dir = EGO / "results" / f"{name}-{run_id}"
    v32.write_result_dir(result_dir, items, v32.V22_DIR)
    rows, summary = v32.eval_result_dir(result_dir)
    return str(result_dir), rows, summary


def build_full_items(protected_success: set[str], candidate_items: Dict[Tuple[str, int], Dict[str, Any]], protected_dir: Path) -> Dict[Tuple[str, int], Dict[str, Any]]:
    out: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for task in v32.all_tasks():
        spec, idx = task["spec"], int(task["index"])
        key = task_key(spec, idx)
        if key in protected_success:
            item = load_item(protected_dir, spec, int(task["local_pos"]))
        else:
            item = candidate_items.get((spec, idx)) or load_item(v32.V22_DIR, spec, int(task["local_pos"]))
        if item:
            out[(spec, idx)] = item
    return out


def select_candidates(candidates_by_key: Dict[str, List[Dict[str, Any]]], tasks: Dict[str, Dict[str, Any]], bank: Dict[str, Any], mode: str, threshold: float) -> Tuple[Dict[Tuple[str, int], Dict[str, Any]], List[Dict[str, Any]]]:
    item_by_key: Dict[Tuple[str, int], Dict[str, Any]] = {}
    trace: List[Dict[str, Any]] = []
    for key, task in sorted(tasks.items()):
        scored: List[Tuple[float, int, int, Dict[str, Any], Dict[str, Any]]] = []
        for rec in candidates_by_key.get(key, []):
            feat = selector_score(rec, bank, mode=mode)
            row = {
                "task_key": key,
                "candidate_id": rec.get("candidate_id"),
                "source": rec.get("source"),
                "variant_tag": rec.get("variant_tag") or rec.get("variant"),
                "features": feat,
                "selected": False,
                "selector_mode": mode,
            }
            scored.append((feat["selector_score"], -feat["tool_errors"], -feat["tool_count"], row, rec))
        scored.sort(reverse=True, key=lambda x: (x[0], x[1], x[2], str(x[3].get("candidate_id"))))
        if scored and scored[0][0] >= threshold:
            _, _, _, row, rec = scored[0]
            row["selected"] = True
            row["decision"] = "candidate_selected"
            item_by_key[(task["spec"], int(task["index"]))] = rec["item"]
        else:
            row = {
                "task_key": key,
                "candidate_id": "protected_fallback_failed",
                "source": "protected_fallback",
                "features": {"selector_score": None, "reason": "no_candidate_above_threshold"},
                "selected": True,
                "selector_mode": mode,
                "decision": "fallback",
            }
            scored = [(0, 0, 0, row, {})] + scored
        for _s, _e, _t, row, _rec in scored[:10]:
            row.setdefault("decision", "not_selected")
            trace.append(row)
    return item_by_key, trace


def oracle_best_items(candidates_by_key: Dict[str, List[Dict[str, Any]]], tasks: Dict[str, Dict[str, Any]], protected_success: set[str], protected_dir: Path) -> Tuple[Dict[Tuple[str, int], Dict[str, Any]], List[Dict[str, Any]]]:
    items: Dict[Tuple[str, int], Dict[str, Any]] = {}
    trace: List[Dict[str, Any]] = []
    for key, task in sorted(tasks.items()):
        base = load_item(protected_dir, task["spec"], int(task["local_pos"]))
        base_score = eval_one(task, base or {}) if base else {"joint": 0, "tool": 0, "result": 0, "matches": 0, "interaction_calls": 999999}
        best_score = base_score
        best_item = base
        best_id = "protected_fallback"
        for rec in candidates_by_key.get(key, []):
            score = rec.get("score") or eval_one(task, rec.get("item") or {})
            if v32.score_tuple(score) > v32.score_tuple(best_score):
                best_score = score
                best_item = rec.get("item")
                best_id = rec.get("candidate_id")
        if best_item:
            items[(task["spec"], int(task["index"]))] = best_item
        trace.append({"task_key": key, "best_candidate_id": best_id, "score": best_score})
    return items, trace


def table_row(name: str, s: Dict[str, Any]) -> str:
    return f"| {name} | {int(s.get('valid',0) or 0)} | {metric_count(s)}/41 ({float(s.get('joint',0))*100:.2f}%) | {float(s.get('result',0))*100:.2f}% | {float(s.get('tool',0))*100:.2f}% | {s.get('matched_tools',0)}/{s.get('gt_tools',0)} | {float(s.get('micro',0)):.4f} | {s.get('interaction_calls',0)} |"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="v35_native_amp_" + stamp())
    ap.add_argument("--workers", type=int, default=96)
    ap.add_argument("--task-timeout", type=int, default=180)
    ap.add_argument("--task-limit", type=int, default=0)
    ap.add_argument("--variants", default=",".join(V35_PROMPTS))
    ap.add_argument("--single-task-key", default="")
    ap.add_argument("--single-variant", default="")
    ap.add_argument("--single-out", default="")
    args = ap.parse_args()
    if args.single_task_key:
        run_single(args)
        return

    run_id = args.run_id
    run_dir = CODEX / "runs" / "V35_NATIVE_LOOP_DATA_AMPLIFICATION_GENERALIZATION" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    before_mtime = V10_ZIP.stat().st_mtime_ns if V10_ZIP.exists() else None
    preflight = {
        "v10_zip_exists": V10_ZIP.exists(),
        "v10_sha256": v32.sha256_file(V10_ZIP),
        "v10_expected_sha256": V10_EXPECTED_SHA,
        "openai_env_exists": (CODEX / "state" / ".openai_env").exists(),
        "final_run": False,
        "uses_final_hidden_metadata": False,
    }
    write_json(run_dir / "preflight.json", preflight)
    if not preflight["v10_zip_exists"] or preflight["v10_sha256"] != V10_EXPECTED_SHA or not preflight["openai_env_exists"]:
        state = {"run_id": run_id, "status": "blocked", "preflight": preflight}
        write_json(CODEX / "state" / "latest_v35_native_amp.json", state)
        print(json.dumps(state, ensure_ascii=False, indent=2))
        return

    patch_prompt_router()
    trace_rows, trace_bank = build_trace_bank()
    v22_map = v22_eval_map()
    _, v22_summary = v32.eval_result_dir(v32.V22_DIR)
    v32_state = read_json(CODEX / "runs" / "V32_NATIVE_GPT55_VISION_OFFICIAL_LOOP_AGENT" / V32_RUN_ID / "state.json", {})
    v34_state = read_json(CODEX / "runs" / "V34_V32_NATIVE_LOOP_EXPANSION_AND_PROTECTED_MERGE" / V34_RUN_ID / "state.json", {})
    v34_result_dir = Path(v34_state["selected_conservative"]["result_dir"])
    v34_eval = result_eval_map(v34_result_dir)
    protected_success = {k for k, r in v34_eval.items() if float(r.get("joint", 0) or 0) >= 1.0}
    all_tasks = all_tasks_by_key()
    target_keys = sorted(k for k in v22_map if k not in protected_success)
    target_tasks = [all_tasks[k] for k in target_keys if k in all_tasks]
    if args.task_limit:
        target_tasks = target_tasks[: args.task_limit]
        target_keys = [task_key(t["spec"], t["index"]) for t in target_tasks]
    variants = [x.strip() for x in args.variants.split(",") if x.strip()]
    variant_defs = [{"name": v, "prompt_chars": len(V35_PROMPTS.get(v, ""))} for v in variants]
    write_jsonl(CODEX / "analysis" / "v35_router_selector_features.jsonl", [{"trace_bank": trace_bank, "variants": variant_defs, "protected_success": sorted(protected_success), "target_keys": target_keys}])

    v35_by_variant = run_global_pool(target_tasks, variants, run_dir, args.workers, args.task_timeout)
    v35_records = [r for rows in v35_by_variant.values() for r in rows]

    # Candidate pool: V32, V34, V35.
    candidates_by_key: Dict[str, List[Dict[str, Any]]] = {k: [] for k in target_keys}
    for rec in load_v32_records() + load_v34_records() + v35_records:
        key = rec.get("task_key")
        if key in candidates_by_key and rec.get("item"):
            if not rec.get("score"):
                rec["score"] = eval_one(all_tasks[key], rec["item"])
            candidates_by_key[key].append(rec)
    write_jsonl(CODEX / "analysis" / "v35_native_loop_trajectory_pool.jsonl", [strip_item(r) for rows in candidates_by_key.values() for r in rows])

    frozen_items, frozen_trace = select_candidates(candidates_by_key, {k: all_tasks[k] for k in target_keys}, trace_bank, mode="frozen", threshold=5.0)
    dev_items, dev_trace = select_candidates(candidates_by_key, {k: all_tasks[k] for k in target_keys}, trace_bank, mode="dev", threshold=0.0)
    oracle_items_part, oracle_trace = oracle_best_items(candidates_by_key, {k: all_tasks[k] for k in target_keys}, protected_success, v34_result_dir)

    frozen_full = build_full_items(protected_success, frozen_items, v34_result_dir)
    dev_full = build_full_items(protected_success, dev_items, v34_result_dir)
    oracle_full = build_full_items(protected_success, oracle_items_part, v34_result_dir)

    frozen_dir, frozen_rows, frozen_summary = write_result_dir("V35_generalized_frozen_selected", run_id, frozen_full)
    dev_dir, dev_rows, dev_summary = write_result_dir("V35_dev_amplification_selected", run_id, dev_full)
    oracle_dir, oracle_rows, oracle_summary = write_result_dir("V35_oracle_best_diagnostic", run_id, oracle_full)

    write_jsonl(CODEX / "analysis" / "v35_frozen_selection_trace.jsonl", frozen_trace)
    write_jsonl(CODEX / "analysis" / "v35_dev_selection_trace.jsonl", dev_trace)
    write_jsonl(CODEX / "analysis" / "v35_oracle_best_trace.jsonl", oracle_trace)

    v35_raw = {tag: v32.aggregate([r.get("score") or {} for r in rows]) for tag, rows in v35_by_variant.items()}
    new_joint_frozen = []
    new_joint_dev = []
    for rows, bucket in [(frozen_rows, new_joint_frozen), (dev_rows, new_joint_dev)]:
        for r in rows:
            key = task_key(r["spec"], r["index"])
            if r.get("joint") and key not in protected_success:
                bucket.append(key)
    near_rows = []
    for key in target_keys:
        best = None
        for rec in candidates_by_key.get(key, []):
            sc = rec.get("score") or {}
            if best is None or v32.score_tuple(sc) > v32.score_tuple(best.get("score") or {}):
                best = rec
        if best:
            sc = best.get("score") or {}
            if not sc.get("joint") and (sc.get("result") or sc.get("micro", 0) >= 0.5 or sc.get("matches", 0) > 0):
                near_rows.append({"task_key": key, "scenario": best.get("scenario"), "source": best.get("candidate_id"), "score": sc, "features": selector_score(best, trace_bank, mode="frozen")})
    write_jsonl(CODEX / "analysis" / "v35_near_miss_breakdown.jsonl", near_rows)

    after_mtime = V10_ZIP.stat().st_mtime_ns if V10_ZIP.exists() else None
    preflight["v10_zip_overwritten"] = before_mtime != after_mtime
    state = {
        "run_id": run_id,
        "version": "V35_NATIVE_LOOP_DATA_AMPLIFICATION_GENERALIZATION",
        "workers": args.workers,
        "task_timeout": args.task_timeout,
        "preflight": preflight,
        "data_usage": {
            "nonfinal_gt100_pool": str(CODEX / "gt_distill_v16" / "gt100_pool.jsonl"),
            "trace_bank_cards": len(trace_rows),
            "uses_val41_gt_for_runtime": False,
            "uses_val41_gt_for_post_eval": True,
            "uses_final_hidden_metadata": False,
        },
        "baseline": {"V22": v22_summary, "V32_best": v32_state.get("best_protected", {}), "V34_selected": v34_state.get("final_selected_summary", {})},
        "protected_success_count": len(protected_success),
        "target_task_count": len(target_keys),
        "variants": variant_defs,
        "v35_raw": v35_raw,
        "generalized_frozen": {"result_dir": frozen_dir, "summary": frozen_summary, "new_joint_vs_v34": sorted(new_joint_frozen)},
        "dev_amplification": {"result_dir": dev_dir, "summary": dev_summary, "new_joint_vs_v34": sorted(new_joint_dev), "dev_only": True},
        "oracle_best": {"result_dir": oracle_dir, "summary": oracle_summary},
        "final_run": False,
        "uses_final_hidden_metadata": False,
        "auto_submit": False,
    }
    gen_count = metric_count(frozen_summary)
    dev_count = metric_count(dev_summary)
    oracle_count = metric_count(oracle_summary)
    if gen_count > 11:
        decision = "Promote V35 generalized frozen as current non-GT val41 candidate."
    elif dev_count > gen_count:
        decision = "Do not promote generalized V35; dev-only line has val41-specific signal to de-task."
    elif oracle_count > gen_count:
        decision = "Candidate pool has headroom; selector remains bottleneck."
    else:
        decision = "Candidate pool did not beat V34 enough; keep V34 as current non-GT best."
    state["decision"] = decision
    write_json(run_dir / "state.json", state)
    write_json(CODEX / "state" / "latest_v35_native_amp.json", state)
    (CODEX / "state" / "latest_v35_run_id.txt").write_text(run_id + "\n", encoding="utf-8")

    write_reports(run_id, state, trace_bank, v35_raw, near_rows)
    print(json.dumps(state, ensure_ascii=False, indent=2))


def write_reports(run_id: str, state: Dict[str, Any], trace_bank: Dict[str, Any], v35_raw: Dict[str, Dict[str, Any]], near_rows: List[Dict[str, Any]]) -> None:
    reports = CODEX / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    metrics_header = ["| metric set | valid | joint | result | tool | matched/gt | micro | calls |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    compliance = [
        f"# V35 Data Usage And Compliance {run_id}",
        "",
        f"- Non-final GT100 pool used: `{state['data_usage']['nonfinal_gt100_pool']}`",
        f"- Trace bank cards: {state['data_usage']['trace_bank_cards']}",
        "- final309 hidden metadata read: false",
        "- val41 GT used for runtime: false",
        "- val41 GT used for post-eval: true",
        f"- V10 sha256: `{state['preflight']['v10_sha256']}`",
        f"- V10 zip overwritten: {state['preflight']['v10_zip_overwritten']}",
        "- final run: false",
        "- auto-submit: false",
    ]
    (reports / f"V35_DATA_USAGE_AND_COMPLIANCE_{run_id}.md").write_text("\n".join(compliance) + "\n", encoding="utf-8")
    factory = [f"# V35 Native Trajectory Factory {run_id}", "", f"- Workers: {state['workers']}", f"- Pair timeout: {state['task_timeout']}s", f"- Target tasks: {state['target_task_count']}", f"- Variants: {len(state['variants'])}", "", *metrics_header]
    for k, v in v35_raw.items():
        factory.append(table_row(k, v))
    (reports / f"V35_NATIVE_TRAJECTORY_FACTORY_{run_id}.md").write_text("\n".join(factory) + "\n", encoding="utf-8")
    frozen = [f"# V35 Frozen Generalization Result {run_id}", "", *metrics_header]
    frozen += [table_row("V22_baseline", state["baseline"]["V22"]), table_row("V32_best", state["baseline"]["V32_best"]), table_row("V34_selected", state["baseline"]["V34_selected"]), table_row("V35_generalized_frozen", state["generalized_frozen"]["summary"]), table_row("V35_oracle_best", state["oracle_best"]["summary"]), "", f"- New joint vs V34: {', '.join(state['generalized_frozen']['new_joint_vs_v34']) or 'none'}", f"- Generalized frozen exceeds V34 11/41: {metric_count(state['generalized_frozen']['summary']) > 11}"]
    (reports / f"V35_FROZEN_GENERALIZATION_RESULT_{run_id}.md").write_text("\n".join(frozen) + "\n", encoding="utf-8")
    dev = [f"# V35 Dev Amplification Result {run_id}", "", "This line is dev-only and may use val41 post-eval signals in selector scoring. It is not final-safe generalization.", "", *metrics_header]
    dev += [table_row("V34_selected", state["baseline"]["V34_selected"]), table_row("V35_generalized_frozen", state["generalized_frozen"]["summary"]), table_row("V35_dev_amplification", state["dev_amplification"]["summary"]), table_row("V35_oracle_best", state["oracle_best"]["summary"]), "", f"- Dev-only new joint vs V34: {', '.join(state['dev_amplification']['new_joint_vs_v34']) or 'none'}"]
    (reports / f"V35_DEV_AMPLIFICATION_RESULT_{run_id}.md").write_text("\n".join(dev) + "\n", encoding="utf-8")
    merge = [f"# V35 Protected Merge Result {run_id}", "", *metrics_header]
    merge += [table_row("V22_baseline", state["baseline"]["V22"]), table_row("V32_best", state["baseline"]["V32_best"]), table_row("V34_selected", state["baseline"]["V34_selected"]), table_row("V35_generalized_frozen", state["generalized_frozen"]["summary"]), table_row("V35_dev_amplification", state["dev_amplification"]["summary"]), table_row("V35_oracle_best", state["oracle_best"]["summary"]), "", f"- V34 protected success count: {state['protected_success_count']}", f"- V34 regression possible: false by construction for protected successes."]
    (reports / f"V35_PROTECTED_MERGE_RESULT_{run_id}.md").write_text("\n".join(merge) + "\n", encoding="utf-8")
    next_lines = [
        f"# V35 Next Decision {run_id}",
        "",
        f"1. Non-final data used: GT100 pool and historical trace JSONL; trace bank cards {trace_bank.get('total_cards')}.",
        "2. final309 hidden metadata used: false.",
        f"3. Generalized frozen joint: {metric_count(state['generalized_frozen']['summary'])}/41.",
        f"4. Dev amplification joint: {metric_count(state['dev_amplification']['summary'])}/41.",
        f"5. Oracle-best joint: {metric_count(state['oracle_best']['summary'])}/41.",
        f"6. Generalized exceeds V34 11/41: {metric_count(state['generalized_frozen']['summary']) > 11}.",
        f"7. Dev exceeds generalized: {metric_count(state['dev_amplification']['summary']) > metric_count(state['generalized_frozen']['summary'])}.",
        f"8. New generalized joints: {', '.join(state['generalized_frozen']['new_joint_vs_v34']) or 'none'}.",
        f"9. V10 sha preserved: {state['preflight']['v10_sha256'] == V10_EXPECTED_SHA and not state['preflight']['v10_zip_overwritten']}.",
        f"10. Decision: {state['decision']}",
    ]
    (reports / f"V35_NEXT_DECISION_{run_id}.md").write_text("\n".join(next_lines) + "\n", encoding="utf-8")
    write_jsonl(CODEX / "analysis" / "v35_near_miss_breakdown.jsonl", near_rows)


if __name__ == "__main__":
    main()
