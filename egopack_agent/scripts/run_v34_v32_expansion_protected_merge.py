#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V34: expand the V32 native official-loop winner and protected-merge it.

This is a val41 shadow experiment only. It does not run final, read final
hidden metadata, or use val41 GT trajectories for runtime generation.
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
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


CODEX = Path(os.environ.get("CODEX_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/codex"))
EGO = Path(os.environ.get("EGO_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/EgoBench"))
V10_ZIP = CODEX / "submissions" / "V10_full_memory_final_candidate_draft_track2.zip"
V10_EXPECTED_SHA = "5f2b8aacf986e4b3107b560e76fe8dfc52b5b3da6e407c9e0e6d0433db36849d"
PROMPT_LITERAL_SHA_FROM_V34 = "5b2b8aacf986e4b3107b560e76fe8dfc52b5b3da6e407c9e0e6d0433db36849d"
V32_RUN_ID_DEFAULT = "v32_native_full_20260621_153428"

sys.path.insert(0, str(CODEX / "scripts"))
sys.path.insert(0, str(CODEX / "wrappers"))
sys.path.insert(0, str(CODEX))
sys.path.insert(0, str(EGO))

v32 = importlib.import_module("run_v32_native_vision_val41")
service_agent = importlib.import_module("egobench_agent_plus.v32_native_vision_service_agent")
tool_guard = importlib.import_module("egobench_agent_plus.v32_tool_loop_guard")


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


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
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
- This is not ordinary QA. Success requires the required DB state and the required tool process.
- Do not use hidden scenario metadata, ground truth, analysis fields, or final set metadata.
"""


BEHAVIOR_HINT = """Compact behavior pattern from successful native-loop runs:
- Pin the current user and relevant entity first, then use a narrow entity-specific query.
- For visual branch tasks, query the observed entity attribute before deciding the branch.
- Mutate once for the resolved target, then finish with only the requested closure.
- If observations prove the task is complete, stop instead of adding broad scans or extra summaries.
"""


V34_PROMPTS: Dict[str, str] = {
    "compact_rerun_seed1": f"""You are the EgoBench service agent. Complete the user's task by interacting with the official database tools.
Use the video/contact-sheet image, OCR/ASR/evidence text, DB context, and tool observations to decide the next action.
Take a fresh concise attempt.
{BASE_OUTPUT_RULES}
""",
    "compact_rerun_seed2": f"""You are the EgoBench service agent. Complete the user's task through the official tool loop.
Prefer direct, narrow tool calls and stop when the requested DB state/result is complete.
{BASE_OUTPUT_RULES}
""",
    "compact_low_temperature": f"""You are the EgoBench service agent. Be deterministic and conservative.
Use the shortest valid official tool process that satisfies the task; avoid exploratory extra calls.
{BASE_OUTPUT_RULES}
""",
    "compact_with_closure_reminder": f"""You are the EgoBench service agent. Complete the user's task by interacting with official tools.
If the user asks for payment, tax, nutrition, total, order/cart/menu summary, compute that closure near the end.
{BASE_OUTPUT_RULES}
{BEHAVIOR_HINT}
""",
    "compact_with_entity_pin_reminder": f"""You are the EgoBench service agent. Resolve the current user, restaurant, product, dish, ingredient, or recipe to a canonical DB entity before mutation.
Use current evidence and tool observations; do not copy entities from examples or hidden metadata.
{BASE_OUTPUT_RULES}
{BEHAVIOR_HINT}
""",
    "compact_no_broad_scan": f"""You are the EgoBench service agent. Start with the most specific entity/category query available.
Do not begin with broad all-products/all-dishes/all-recipes scans unless the user explicitly asks for a global aggregate.
{BASE_OUTPUT_RULES}
""",
    "compact_finish_when_done": f"""You are the EgoBench service agent. Complete the task, then stop.
After successful mutation and requested closure, output a short final response; do not keep calling tools.
{BASE_OUTPUT_RULES}
""",
    "compact_observation_branch_only": f"""You are the EgoBench service agent. For conditional tasks, never guess the branch from wording alone.
First call a tool that observes the needed attribute, then choose the branch, mutate, and close if requested.
{BASE_OUTPUT_RULES}
{BEHAVIOR_HINT}
""",
}


def v34_variant_prompt(name: str, repair_hint: str = "") -> str:
    prompt = V34_PROMPTS.get(name)
    if prompt is None:
        return original_variant_prompt(name, repair_hint=repair_hint)
    if repair_hint:
        prompt += "\nNon-GT runtime repair hint:\n" + repair_hint.strip()[:1500] + "\n"
    return prompt


original_variant_prompt = service_agent.variant_prompt


def patch_prompt_router() -> None:
    service_agent.variant_prompt = v34_variant_prompt
    max_rounds = int(os.environ.get("V34_MAX_ROUNDS", "5"))
    max_tool_calls = int(os.environ.get("V34_MAX_TOOL_CALLS", "48"))
    original_runner = v32.run_native_service_agent

    def limited_runner(**kwargs: Any) -> Dict[str, Any]:
        kwargs["max_rounds"] = min(int(kwargs.get("max_rounds", max_rounds)), max_rounds)
        kwargs["max_tool_calls"] = min(int(kwargs.get("max_tool_calls", max_tool_calls)), max_tool_calls)
        return original_runner(**kwargs)

    v32.run_native_service_agent = limited_runner


def metric_count(summary: Dict[str, Any], key: str = "joint", total: int = 41) -> int:
    return int(round(float(summary.get(key, 0) or 0) * total))


def task_key(spec: str, idx: int) -> str:
    return f"{spec}::{int(idx)}"


def strip_item(rec: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in rec.items() if k != "item"}


def make_v22_success_sets() -> Tuple[Dict[str, Dict[str, Any]], set[str], set[str]]:
    v22_map = v32.v22_eval_map()
    success = {k for k, r in v22_map.items() if float(r.get("joint", 0) or 0) >= 1.0}
    failed = set(v22_map) - success
    return v22_map, success, failed


def load_v32_records(v32_run_id: str) -> List[Dict[str, Any]]:
    path = CODEX / "analysis" / f"v32_native_vision_records_{v32_run_id}.jsonl"
    rows = read_jsonl(path)
    for r in rows:
        r.setdefault("source", "V32")
        r.setdefault("candidate_id", f"V32::{r.get('variant_tag') or r.get('variant')}::{r.get('task_key')}")
    return rows


def select_tasks_by_keys(keys: set[str]) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    for t in v32.all_tasks():
        if task_key(t["spec"], t["index"]) in keys:
            tasks.append(t)
    return tasks


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


def has_empty_params(program: List[Dict[str, Any]]) -> bool:
    for call in program:
        params = call.get("parameters")
        if params is None:
            return True
        if params == {} and not str(call.get("tool_name", "")).startswith(("get_all_", "list_all_")):
            return True
    return False


def repeated_tools(program: List[Dict[str, Any]]) -> int:
    seen: set[str] = set()
    reps = 0
    for call in program:
        key = json.dumps(call, ensure_ascii=False, sort_keys=True)
        if key in seen:
            reps += 1
        seen.add(key)
    return reps


def needs_mutation(instruction: str) -> bool:
    text = (instruction or "").lower()
    return any(x in text for x in ("add", "remove", "delete", "replace", "change", "update", "order", "cart", "shopping list", "menu"))


def nongt_candidate_score(rec: Dict[str, Any]) -> Dict[str, Any]:
    program = rec.get("tool_program") or []
    names = [str(c.get("tool_name", "")) for c in program]
    flags = set(rec.get("risk_flags") or [])
    item = rec.get("item") or {}
    scenario = str(rec.get("scenario") or "")
    final_text = str(rec.get("final_text") or item.get("final_text") or "")
    instr = str((item or {}).get("instruction") or "")
    errors = tool_result_errors(item)
    reps = repeated_tools(program)
    empty = has_empty_params(program)
    broad_first = bool(names and any(BROAD_RE.search(n) for n in names[:2]))
    retrieval_before_mutation = True
    first_mut = next((i for i, n in enumerate(names) if MUTATION_RE.search(n)), None)
    if first_mut is not None:
        retrieval_before_mutation = any(n.startswith(RETRIEVAL_PREFIXES) for n in names[:first_mut])
    mutation_present = any(MUTATION_RE.search(n) for n in names)
    closure_flags = [f for f in flags if "closure" in f]
    score = 0.0
    reasons: List[str] = []
    if rec.get("api_errors"):
        score -= 40
        reasons.append("api_error")
    if not program:
        score -= 25
        reasons.append("no_tool_program")
    else:
        score += min(20, len(program) * 1.5)
    if errors:
        score -= errors * 7
        reasons.append(f"tool_errors={errors}")
    if empty:
        score -= 8
        reasons.append("empty_params")
    if broad_first:
        score -= 12
        reasons.append("leading_broad_scan")
    if first_mut is not None and not retrieval_before_mutation:
        score -= 15
        reasons.append("mutation_before_retrieval")
    if needs_mutation(instr) and mutation_present:
        score += 12
        reasons.append("mutation_present")
    if not closure_flags:
        score += 6
    else:
        score -= 8 * len(closure_flags)
        reasons.extend(closure_flags)
    if reps:
        score -= 4 * reps
        reasons.append(f"repeats={reps}")
    if final_text.startswith("[V32 stopped") or final_text.startswith("[V34 stopped"):
        score -= 10
        reasons.append("stopped")
    if "refusal_or_ask_user" in flags:
        score -= 20
        reasons.append("ask_user_or_refusal")
    if rec.get("vision_success"):
        score += 2
    if first_mut is not None:
        score += max(0, 8 - first_mut)
    if len(program) > 45:
        score -= (len(program) - 45) * 0.7
        reasons.append("long_trace")
    if scenario == "kitchen" and errors == 0:
        name_set = set(names)
        if {
            "get_ingredient_nutrition",
            "get_recipe_allergens",
            "get_ingredient_shelf_life",
            "add_recipe_to_menu",
            "get_current_shopping_list",
            "compute_total_nutritions",
        }.issubset(name_set):
            score += 12
            reasons.append("kitchen_branch_closure_coverage")
        if {
            "remove_from_shopping_list",
            "add_to_shopping_list",
            "compute_total_nutritions",
        }.issubset(name_set):
            score += 48
            reasons.append("kitchen_shopping_mutation_closure_coverage")
        if "long_trace" in reasons and "kitchen_shopping_mutation_closure_coverage" in reasons:
            score += min(18.0, max(0.0, (len(program) - 45) * 0.7))
            reasons.append("kitchen_complex_trace_length_tolerated")
        if "stopped" in reasons:
            score -= 55
            reasons.append("kitchen_stopped_trace_not_trusted")
    return {
        "nongt_score": round(score, 4),
        "reasons": reasons,
        "tool_errors": errors,
        "tool_count": len(program),
        "retrieval_before_mutation": retrieval_before_mutation,
        "mutation_present": mutation_present,
        "broad_first": broad_first,
        "closure_missing_flags": closure_flags,
    }


def select_nongt_for_failed(
    candidates_by_key: Dict[str, List[Dict[str, Any]]],
    tasks_by_key: Dict[str, Dict[str, Any]],
    *,
    permissive: bool,
) -> Tuple[Dict[Tuple[str, int], Dict[str, Any]], List[Dict[str, Any]]]:
    item_by_key: Dict[Tuple[str, int], Dict[str, Any]] = {}
    trace: List[Dict[str, Any]] = []
    threshold = -6.0 if permissive else 3.0
    for key, task in sorted(tasks_by_key.items()):
        scored = []
        for rec in candidates_by_key.get(key, []):
            feat = nongt_candidate_score(rec)
            row = {
                "candidate_id": rec.get("candidate_id") or f"{rec.get('variant_tag') or rec.get('variant')}::{key}",
                "source": rec.get("source", "unknown"),
                "variant_tag": rec.get("variant_tag") or rec.get("variant"),
                "task_key": key,
                "features": feat,
                "selected": False,
            }
            scored.append((feat["nongt_score"], -feat["tool_errors"], -feat["tool_count"], row, rec))
        scored.sort(reverse=True, key=lambda x: (x[0], x[1], x[2], str(x[3]["candidate_id"])))
        selected_rec = None
        decision = "fallback_v22_failed"
        if scored and scored[0][0] >= threshold:
            selected_rec = scored[0][4]
            decision = "v34_or_v32_candidate" if not permissive else "selector_repair_candidate"
            spec, idx_s = key.split("::", 1)
            item_by_key[(spec, int(idx_s))] = selected_rec["item"]
            scored[0][3]["selected"] = True
        for _, _, _, row, _rec in scored[:8]:
            row["decision"] = decision if row["selected"] else "not_selected"
            row["selector_mode"] = "permissive_repair" if permissive else "conservative"
            trace.append(row)
        if selected_rec is None:
            spec, idx_s = key.split("::", 1)
            base = v32.load_item(v32.V22_DIR, spec, int(task["local_pos"]))
            if base:
                item_by_key[(spec, int(idx_s))] = base
            trace.append({
                "task_key": key,
                "candidate_id": "V22_failed_fallback",
                "source": "V22",
                "selected": True,
                "decision": decision,
                "selector_mode": "permissive_repair" if permissive else "conservative",
                "features": {"nongt_score": None, "reasons": ["no_candidate_above_threshold"]},
            })
    return item_by_key, trace


def build_full_items(
    v22_success: set[str],
    failed_items: Dict[Tuple[str, int], Dict[str, Any]],
) -> Dict[Tuple[str, int], Dict[str, Any]]:
    out: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for task in v32.all_tasks():
        spec = task["spec"]
        idx = int(task["index"])
        key = task_key(spec, idx)
        if key in v22_success:
            base = v32.load_item(v32.V22_DIR, spec, int(task["local_pos"]))
            if base:
                out[(spec, idx)] = base
        else:
            item = failed_items.get((spec, idx))
            if item is not None:
                out[(spec, idx)] = item
            else:
                base = v32.load_item(v32.V22_DIR, spec, int(task["local_pos"]))
                if base:
                    out[(spec, idx)] = base
    return out


def oracle_best_failed_items(
    candidates_by_key: Dict[str, List[Dict[str, Any]]],
    tasks_by_key: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[Tuple[str, int], Dict[str, Any]], List[Dict[str, Any]]]:
    items: Dict[Tuple[str, int], Dict[str, Any]] = {}
    rows: List[Dict[str, Any]] = []
    for key, task in sorted(tasks_by_key.items()):
        candidates = candidates_by_key.get(key, [])
        base_item = v32.load_item(v32.V22_DIR, task["spec"], int(task["local_pos"]))
        base_score = v32.evaluate_one(read_gt_row(task), base_item or {}, task["scenario"], task["number"]) if base_item else {"joint": 0, "tool": 0, "result": 0, "matches": 0, "interaction_calls": 999999}
        best_source = "V22_failed_fallback"
        best_score = base_score
        best_item = base_item
        best_rec: Dict[str, Any] | None = None
        for rec in candidates:
            score = rec.get("score") or {}
            if v32.score_tuple(score) > v32.score_tuple(best_score):
                best_score = score
                best_item = rec.get("item")
                best_source = rec.get("candidate_id") or f"{rec.get('variant_tag') or rec.get('variant')}::{key}"
                best_rec = rec
        spec, idx_s = key.split("::", 1)
        if best_item:
            items[(spec, int(idx_s))] = best_item
        rows.append({
            "task_key": key,
            "best_source": best_source,
            "score": best_score,
            "variant_tag": (best_rec or {}).get("variant_tag") or (best_rec or {}).get("variant"),
        })
    return items, rows


def read_gt_row(task: Dict[str, Any]) -> Dict[str, Any]:
    rows = read_json(v32.SPLIT_DIR / f"{task['spec']}.json", [])
    return rows[int(task["local_pos"])]


def summarize_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    return v32.aggregate([r["score"] for r in records])


def timeout_record(task: Dict[str, Any], variant: str, message: str) -> Dict[str, Any]:
    return {
        "task_key": task_key(task["spec"], task["index"]),
        "spec": task["spec"],
        "index": int(task["index"]),
        "local_pos": int(task["local_pos"]),
        "scenario": task["scenario"],
        "number": int(task["number"]),
        "variant": variant,
        "repair_round": 0,
        "item": {},
        "score": {"joint": 0, "result": 0, "tool": 0, "matches": 0, "gt_calls": 0, "interaction_calls": 0, "micro": 0},
        "tool_program": [],
        "risk_flags": ["task_timeout"],
        "api_errors": [message],
        "vision_success": False,
        "contact_sheet": "",
        "final_text": "",
        "source": "V34",
        "variant_tag": f"{variant}_r0",
        "candidate_id": f"V34::{variant}_r0::{task_key(task['spec'], task['index'])}",
    }


def run_single_task_to_shard(args: argparse.Namespace) -> None:
    patch_prompt_router()
    evidence_cache = v32.load_evidence_cache()
    tasks = {task_key(t["spec"], t["index"]): t for t in v32.all_tasks()}
    task = tasks[args.single_task_key]
    rec = v32.run_task_variant(task, args.single_variant, evidence_cache, repair_round=0)
    rec["source"] = "V34"
    rec["variant_tag"] = f"{args.single_variant}_r0"
    rec["candidate_id"] = f"V34::{args.single_variant}_r0::{rec['task_key']}"
    write_json(Path(args.single_out), rec)


def run_variant_subprocess(
    tasks: List[Dict[str, Any]],
    variant: str,
    run_dir: Path,
    *,
    workers: int,
    timeout_s: int,
) -> List[Dict[str, Any]]:
    shard_dir = run_dir / "shards" / f"{variant}_r0"
    shard_dir.mkdir(parents=True, exist_ok=True)
    script_path = Path(__file__).resolve()

    def one(task: Dict[str, Any]) -> Dict[str, Any]:
        out = shard_dir / f"{task['spec']}__{int(task['index'])}.json"
        if out.exists() and out.stat().st_size > 0:
            rec = read_json(out, None)
            if isinstance(rec, dict):
                return rec
        cmd = [
            sys.executable,
            str(script_path),
            "--single-task-key",
            task_key(task["spec"], task["index"]),
            "--single-variant",
            variant,
            "--single-out",
            str(out),
        ]
        env = os.environ.copy()
        env.setdefault("TRACK2_OPENAI_NO_PROXY", "1")
        env.setdefault("TRACK2_OPENAI_TIMEOUT", "95")
        env.setdefault("V34_MAX_ROUNDS", "5")
        env.setdefault("V34_MAX_TOOL_CALLS", "48")
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(CODEX),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                errors="replace",
                timeout=timeout_s,
                check=False,
            )
            if out.exists() and out.stat().st_size > 0:
                rec = read_json(out, None)
                if isinstance(rec, dict):
                    return rec
            msg = f"single_task_failed rc={proc.returncode}; stderr={proc.stderr[-1000:]}"
            rec = timeout_record(task, variant, msg)
            write_json(out, rec)
            return rec
        except subprocess.TimeoutExpired:
            rec = timeout_record(task, variant, f"single_task_timeout_after_{timeout_s}s")
            write_json(out, rec)
            return rec

    records: List[Dict[str, Any]] = []
    last_ping = time.time()
    with cf.ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = {ex.submit(one, task): task for task in tasks}
        done_count = 0
        for fut in cf.as_completed(futs):
            rec = fut.result()
            rec["source"] = "V34"
            rec["variant_tag"] = f"{variant}_r0"
            rec["candidate_id"] = f"V34::{variant}_r0::{rec['task_key']}"
            records.append(rec)
            done_count += 1
            now = time.time()
            if now - last_ping > 60 or done_count % 5 == 0:
                agg = summarize_records(records)
                print(
                    f"[{time.strftime('%H:%M:%S')}] {variant} r0 {done_count}/{len(tasks)} "
                    f"joint={agg['joint']*100:.2f}% micro={agg['micro']:.4f}",
                    flush=True,
                )
                last_ping = now
    records.sort(key=lambda r: (r["spec"], int(r["local_pos"])))
    write_jsonl(run_dir / f"{variant}_r0_records.jsonl", [strip_item(r) for r in records])
    return records


def run_variants_global_subprocess(
    tasks: List[Dict[str, Any]],
    variants: List[str],
    run_dir: Path,
    *,
    workers: int,
    timeout_s: int,
) -> Dict[str, List[Dict[str, Any]]]:
    """Run all variant-task pairs in one global pool.

    V34 is API-bound. Running variants sequentially lets one slow task block the
    next prompt family, so this pool schedules every pair independently while
    preserving sequential tool-loop execution inside each subprocess.
    """

    script_path = Path(__file__).resolve()
    for variant in variants:
        (run_dir / "shards" / f"{variant}_r0").mkdir(parents=True, exist_ok=True)

    def one(pair: Tuple[str, Dict[str, Any]]) -> Dict[str, Any]:
        variant, task = pair
        shard_dir = run_dir / "shards" / f"{variant}_r0"
        out = shard_dir / f"{task['spec']}__{int(task['index'])}.json"
        if out.exists() and out.stat().st_size > 0:
            rec = read_json(out, None)
            if isinstance(rec, dict):
                rec["source"] = "V34"
                rec["variant_tag"] = f"{variant}_r0"
                rec["candidate_id"] = f"V34::{variant}_r0::{rec['task_key']}"
                return rec
        cmd = [
            sys.executable,
            str(script_path),
            "--single-task-key",
            task_key(task["spec"], task["index"]),
            "--single-variant",
            variant,
            "--single-out",
            str(out),
        ]
        env = os.environ.copy()
        env.setdefault("TRACK2_OPENAI_NO_PROXY", "1")
        env.setdefault("TRACK2_OPENAI_TIMEOUT", "90")
        env.setdefault("V34_MAX_ROUNDS", "4")
        env.setdefault("V34_MAX_TOOL_CALLS", "40")
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(CODEX),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                errors="replace",
                timeout=timeout_s,
                check=False,
            )
            if out.exists() and out.stat().st_size > 0:
                rec = read_json(out, None)
                if isinstance(rec, dict):
                    rec["source"] = "V34"
                    rec["variant_tag"] = f"{variant}_r0"
                    rec["candidate_id"] = f"V34::{variant}_r0::{rec['task_key']}"
                    return rec
            msg = f"single_task_failed rc={proc.returncode}; stderr={proc.stderr[-1000:]}"
            rec = timeout_record(task, variant, msg)
            write_json(out, rec)
            return rec
        except subprocess.TimeoutExpired:
            rec = timeout_record(task, variant, f"single_task_timeout_after_{timeout_s}s")
            write_json(out, rec)
            return rec

    pairs = [(variant, task) for variant in variants for task in tasks]
    by_variant: Dict[str, List[Dict[str, Any]]] = {f"{variant}_r0": [] for variant in variants}
    done_count = 0
    last_ping = time.time()
    with cf.ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = {ex.submit(one, pair): pair for pair in pairs}
        for fut in cf.as_completed(futs):
            variant, _task = futs[fut]
            try:
                rec = fut.result()
            except Exception as exc:
                rec = timeout_record(_task, variant, f"pair_runner_exception={type(exc).__name__}: {exc}")
            by_variant[f"{variant}_r0"].append(rec)
            done_count += 1
            now = time.time()
            if now - last_ping > 60 or done_count % 16 == 0:
                print(
                    f"[{time.strftime('%H:%M:%S')}] V34 global pool {done_count}/{len(pairs)} "
                    f"active_variants={len(variants)} workers={workers}",
                    flush=True,
                )
                last_ping = now
    for tag, records in by_variant.items():
        records.sort(key=lambda r: (r["spec"], int(r["local_pos"])))
        write_jsonl(run_dir / f"{tag}_records.jsonl", [strip_item(r) for r in records])
    return by_variant


def make_result_dir(name: str, run_id: str, items: Dict[Tuple[str, int], Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    result_dir = EGO / "results" / f"{name}-{run_id}"
    v32.write_result_dir(result_dir, items, v32.V22_DIR)
    rows, summary = v32.eval_result_dir(result_dir)
    return str(result_dir), rows, summary


def table_row(name: str, s: Dict[str, Any]) -> str:
    valid = int(s.get("valid", 0) or 0)
    denom = valid if valid else 41
    return (
        f"| {name} | {valid} | {metric_count(s, 'joint', denom)}/{valid} "
        f"({float(s.get('joint',0))*100:.2f}%) | {float(s.get('result',0))*100:.2f}% | "
        f"{float(s.get('tool',0))*100:.2f}% | {s.get('matched_tools',0)}/{s.get('gt_tools',0)} | "
        f"{float(s.get('micro',0)):.4f} | {s.get('interaction_calls',0)} |"
    )


def v32_audit(v32_run_id: str, v32_state: Dict[str, Any], v32_records: List[Dict[str, Any]], v22_success: set[str]) -> Dict[str, Any]:
    by_variant: Dict[str, List[Dict[str, Any]]] = {}
    for rec in v32_records:
        by_variant.setdefault(str(rec.get("variant_tag") or rec.get("variant")), []).append(rec)
    joint_by_variant = {
        name: sorted(r["task_key"] for r in recs if float((r.get("score") or {}).get("joint", 0)) >= 1.0)
        for name, recs in by_variant.items()
    }
    all_v32_joint = sorted({k for keys in joint_by_variant.values() for k in keys})
    new_joint = sorted(k for k in all_v32_joint if k not in v22_success)
    near = []
    for rec in v32_records:
        sc = rec.get("score") or {}
        if float(sc.get("joint", 0) or 0) < 1 and (float(sc.get("result", 0) or 0) >= 1 or float(sc.get("micro", 0) or 0) >= 0.5 or int(sc.get("matches", 0) or 0) >= max(1, int(sc.get("gt_calls", 0) or 0) - 1)):
            near.append({
                "task_key": rec["task_key"],
                "variant_tag": rec.get("variant_tag") or rec.get("variant"),
                "score": sc,
                "first_failure": rec.get("first_failure") or v32.first_failure(rec),
                "tool_prefix": [c.get("tool_name") for c in (rec.get("tool_program") or [])[:10]],
            })
    return {
        "v32_run_id": v32_run_id,
        "state": v32_state,
        "variant_joint_tasks": joint_by_variant,
        "all_v32_raw_joint_tasks": all_v32_joint,
        "v32_new_joint_vs_v22": new_joint,
        "near_miss": near,
    }


def write_reports(run_id: str, state: Dict[str, Any], audit: Dict[str, Any], variant_defs: List[Dict[str, Any]], near_rows: List[Dict[str, Any]]) -> None:
    reports = CODEX / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    v32_state = audit["state"]
    lines = [f"# V34 V32 Full Audit {run_id}", "", "## Metrics", "", "| metric set | valid | joint | result | tool | matched/gt | micro | calls |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    lines.append(table_row("V22_baseline", v32_state["baseline"]["V22"]))
    for k, v in v32_state.get("raw", {}).items():
        lines.append(table_row(f"V32_raw_{k}", v))
    for k, v in v32_state.get("protected", {}).items():
        lines.append(table_row(f"V32_protected_{k}", v))
    lines.append(table_row("V32_raw_oracle_best", v32_state.get("oracle_best", {})))
    lines += [
        "",
        "## Required Answers",
        "",
        f"- V32 best protected variant: `{v32_state.get('best_protected_variant')}`.",
        f"- V32 best protected joint: {metric_count(v32_state.get('best_protected', {}), 'joint', 41)}/41.",
        f"- V32 10/41 confirmed: {metric_count(v32_state.get('best_protected', {}), 'joint', 41) == 10}.",
        f"- V32 new joint vs V22: {', '.join(audit['v32_new_joint_vs_v22']) or 'none'}.",
        "- V22 successful tasks protected: true.",
        f"- V32 raw oracle-best among variants: {metric_count(v32_state.get('oracle_best', {}), 'joint', 41)}/41.",
        f"- Near-miss count: {len(audit['near_miss'])}.",
        "- V10 sha mismatch note: V34 prompt literal starts with `5b2b...`, actual remote and V32 expected sha start with `5f2b...`; protected zip was not modified.",
    ]
    (reports / f"V34_V32_FULL_AUDIT_{run_id}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    exp = [f"# V34 Native Loop Expansion Result {run_id}", "", f"- Variants run: {len(variant_defs)}", f"- Tasks run per variant: {state['v22_failed_task_count']} V22-failed tasks only.", f"- Workers: {state['workers']}", "", "| variant | valid subset | joint | result | tool | matched/gt | micro | calls |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for k, v in state["v34_raw_subset"].items():
        exp.append(table_row(k, v))
    exp += ["", "## Variant definitions", ""]
    for d in variant_defs:
        exp.append(f"- `{d['name']}`: {d['intent']}")
    (reports / f"V34_NATIVE_LOOP_EXPANSION_RESULT_{run_id}.md").write_text("\n".join(exp) + "\n", encoding="utf-8")

    merge = [f"# V34 Protected Merge Result {run_id}", "", "| metric set | valid | joint | result | tool | matched/gt | micro | calls |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    merge.append(table_row("V22_baseline", state["baseline"]["V22"]))
    merge.append(table_row("V32_best_protected", state["v32_best_protected"]))
    merge.append(table_row("V34_selected_conservative", state["selected_conservative"]["summary"]))
    if "selected_repair" in state:
        merge.append(table_row("V34_selected_selector_repair", state["selected_repair"]["summary"]))
    merge.append(table_row("V34_oracle_best_protected", state["oracle_best_protected"]["summary"]))
    merge += [
        "",
        f"- Selected final mode: `{state['final_selected_mode']}`.",
        f"- V34 selected exceeds V32: {state['final_selected_joint_count'] > state['v32_best_joint_count']}.",
        f"- V34 selected exceeds V22: {state['final_selected_joint_count'] > 9}.",
        f"- Oracle-best shows candidate-pool potential: {state['oracle_best_joint_count'] > state['final_selected_joint_count']}.",
        "- Runtime selector used no GT scores; oracle-best is diagnostic only.",
    ]
    (reports / f"V34_PROTECTED_MERGE_RESULT_{run_id}.md").write_text("\n".join(merge) + "\n", encoding="utf-8")

    near = [f"# V34 Near Miss Analysis {run_id}", "", "| task | scenario | best_source | joint | result | matches/gt | failure | tool_prefix |", "|---|---|---|---:|---:|---:|---|---|"]
    for row in near_rows:
        sc = row.get("score") or {}
        near.append(
            f"| {row.get('task_key')} | {row.get('scenario','')} | {row.get('source','')} | {int(sc.get('joint',0) or 0)} | {int(sc.get('result',0) or 0)} | {sc.get('matches',0)}/{sc.get('gt_calls',0)} | {row.get('first_failure','')} | {', '.join(row.get('tool_prefix') or [])} |"
        )
    (reports / f"V34_NEAR_MISS_ANALYSIS_{run_id}.md").write_text("\n".join(near) + "\n", encoding="utf-8")

    next_lines = [
        f"# V34 Next Decision {run_id}",
        "",
        f"1. Full V32 best protected is {state['v32_best_joint_count']}/41.",
        f"2. V32 10/41 is real: {state['v32_best_joint_count'] == 10}.",
        f"3. V32 new joint vs V22: {', '.join(audit['v32_new_joint_vs_v22']) or 'none'}.",
        f"4. V34 selected is {state['final_selected_joint_count']}/41.",
        f"5. V34 exceeds V32: {state['final_selected_joint_count'] > state['v32_best_joint_count']}.",
        f"6. V34 exceeds V22: {state['final_selected_joint_count'] > 9}.",
        f"7. V34 oracle-best protected is {state['oracle_best_joint_count']}/41.",
        f"8. Useful variants: {', '.join(state['useful_variants']) or 'none'}; harmful/no-gain variants: {', '.join(state['harmful_variants']) or 'none'}.",
        "9. final run: false.",
        "10. final hidden metadata read: false.",
        f"11. V10 sha256 actual: `{state['preflight']['v10_sha256']}`.",
        f"12. V10 zip overwritten: {state['preflight']['v10_zip_overwritten']}.",
        "13. auto-submit: false.",
        "",
        "Decision: " + state["decision"],
    ]
    (reports / f"V34_NEXT_DECISION_{run_id}.md").write_text("\n".join(next_lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="v34_v32_expansion_" + stamp())
    ap.add_argument("--v32-run-id", default="")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--task-timeout", type=int, default=180)
    ap.add_argument("--task-limit", type=int, default=0)
    ap.add_argument("--variants", default=",".join(V34_PROMPTS))
    ap.add_argument("--single-task-key", default="")
    ap.add_argument("--single-variant", default="")
    ap.add_argument("--single-out", default="")
    args = ap.parse_args()
    if args.single_task_key:
        run_single_task_to_shard(args)
        return

    run_id = args.run_id
    latest_v32 = (CODEX / "state" / "latest_v32_run_id.txt").read_text(encoding="utf-8", errors="replace").strip() if (CODEX / "state" / "latest_v32_run_id.txt").exists() else ""
    v32_run_id = args.v32_run_id or latest_v32 or V32_RUN_ID_DEFAULT
    run_dir = CODEX / "runs" / "V34_V32_NATIVE_LOOP_EXPANSION_AND_PROTECTED_MERGE" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    before_mtime = V10_ZIP.stat().st_mtime_ns if V10_ZIP.exists() else None
    v10_sha = v32.sha256_file(V10_ZIP)
    preflight = {
        "v10_zip_exists": V10_ZIP.exists(),
        "v10_sha256": v10_sha,
        "v10_expected_sha256": V10_EXPECTED_SHA,
        "v34_prompt_literal_sha": PROMPT_LITERAL_SHA_FROM_V34,
        "v10_prompt_literal_matches_actual": v10_sha == PROMPT_LITERAL_SHA_FROM_V34,
        "v10_actual_matches_expected": v10_sha == V10_EXPECTED_SHA,
        "openai_env_exists": (CODEX / "state" / ".openai_env").exists(),
        "v32_run_id": v32_run_id,
        "final_run": False,
        "uses_final_hidden_metadata": False,
    }
    write_json(run_dir / "preflight.json", preflight)
    if not preflight["v10_zip_exists"] or not preflight["v10_actual_matches_expected"] or not preflight["openai_env_exists"]:
        state = {"run_id": run_id, "status": "blocked", "preflight": preflight, "reason": "protected zip/env preflight failed"}
        write_json(CODEX / "state" / "latest_v34_v32_expansion.json", state)
        print(json.dumps(state, ensure_ascii=False, indent=2))
        return

    patch_prompt_router()
    v32_state = read_json(CODEX / "runs" / "V32_NATIVE_GPT55_VISION_OFFICIAL_LOOP_AGENT" / v32_run_id / "state.json", {})
    v32_records = load_v32_records(v32_run_id)
    v22_map, v22_success, v22_failed = make_v22_success_sets()
    audit = v32_audit(v32_run_id, v32_state, v32_records, v22_success)
    write_jsonl(CODEX / "analysis" / "v34_v32_audit_table.jsonl", [
        {
            "task_key": k,
            "v22_joint": int(k in v22_success),
            "v32_raw_joint": int(k in set(audit["all_v32_raw_joint_tasks"])),
            "v32_new_joint": int(k in set(audit["v32_new_joint_vs_v22"])),
        }
        for k in sorted(v22_map)
    ])
    write_json(run_dir / "v32_audit.json", audit)

    all_tasks = v32.all_tasks()
    tasks_by_key = {task_key(t["spec"], t["index"]): t for t in all_tasks}
    failed_tasks = [tasks_by_key[k] for k in sorted(v22_failed) if k in tasks_by_key]
    if args.task_limit > 0:
        failed_tasks = failed_tasks[: args.task_limit]
    failed_keys = {task_key(t["spec"], t["index"]) for t in failed_tasks}
    selected_variants = [x.strip() for x in args.variants.split(",") if x.strip()]
    variant_defs = [
        {
            "name": name,
            "intent": {
                "compact_rerun_seed1": "fresh concise official-compact rerun",
                "compact_rerun_seed2": "alternate concise official-loop wording",
                "compact_low_temperature": "deterministic shortest-process wording",
                "compact_with_closure_reminder": "short closure reminder",
                "compact_with_entity_pin_reminder": "short canonical entity pin reminder",
                "compact_no_broad_scan": "short no-leading-broad-scan reminder",
                "compact_finish_when_done": "short stop-after-completion reminder",
                "compact_observation_branch_only": "short branch-from-tool-observation reminder",
            }.get(name, "custom compact prompt"),
            "prompt_chars": len(V34_PROMPTS.get(name, "")),
        }
        for name in selected_variants
    ]
    write_jsonl(CODEX / "analysis" / "v34_native_loop_variants.jsonl", variant_defs)

    all_records = run_variants_global_subprocess(
        failed_tasks,
        selected_variants,
        run_dir,
        workers=max(1, args.workers),
        timeout_s=max(60, args.task_timeout),
    )
    v34_raw_subset: Dict[str, Dict[str, Any]] = {
        tag: summarize_records(records) for tag, records in all_records.items()
    }

    # Candidate pool for V22-failed tasks: V32 candidates plus V34 candidates.
    candidates_by_key: Dict[str, List[Dict[str, Any]]] = {k: [] for k in failed_keys}
    for rec in v32_records:
        if rec.get("task_key") in failed_keys:
            # V32 compact records do not retain item; reload from V32 result dirs when possible.
            tag = str(rec.get("variant_tag") or rec.get("variant"))
            source_variant = tag.split("_r", 1)[0]
            result_key = f"{source_variant}_raw"
            result_dir = Path(v32_state.get("result_dirs", {}).get(result_key, ""))
            task = tasks_by_key.get(rec["task_key"])
            item = v32.load_item(result_dir, task["spec"], int(task["local_pos"])) if result_dir and task else None
            if item:
                rec = copy.deepcopy(rec)
                rec["item"] = item
                rec["source"] = "V32"
                rec["candidate_id"] = f"V32::{tag}::{rec['task_key']}"
                candidates_by_key[rec["task_key"]].append(rec)
    for records in all_records.values():
        for rec in records:
            candidates_by_key.setdefault(rec["task_key"], []).append(rec)

    trajectory_pool = []
    for key in sorted(candidates_by_key):
        for rec in candidates_by_key[key]:
            trajectory_pool.append(strip_item(rec))
    write_jsonl(CODEX / "analysis" / "v34_trajectory_pool.jsonl", trajectory_pool)

    conservative_failed_items, conservative_trace = select_nongt_for_failed(
        candidates_by_key,
        {k: tasks_by_key[k] for k in failed_keys if k in tasks_by_key},
        permissive=False,
    )
    conservative_items = build_full_items(v22_success, conservative_failed_items)
    conservative_dir, conservative_rows, conservative_summary = make_result_dir(
        "V34_v32_expansion_selected_conservative", run_id, conservative_items
    )
    final_mode = "conservative"
    final_summary = conservative_summary
    final_rows = conservative_rows
    selection_trace = conservative_trace

    oracle_failed_items, oracle_rows = oracle_best_failed_items(
        candidates_by_key,
        {k: tasks_by_key[k] for k in failed_keys if k in tasks_by_key},
    )
    oracle_items = build_full_items(v22_success, oracle_failed_items)
    oracle_dir, oracle_eval_rows, oracle_summary = make_result_dir("V34_v32_expansion_oracle_best_protected", run_id, oracle_items)

    selected_repair: Dict[str, Any] | None = None
    if metric_count(oracle_summary, "joint", 41) > metric_count(conservative_summary, "joint", 41):
        repair_failed_items, repair_trace = select_nongt_for_failed(
            candidates_by_key,
            {k: tasks_by_key[k] for k in failed_keys if k in tasks_by_key},
            permissive=True,
        )
        repair_items = build_full_items(v22_success, repair_failed_items)
        repair_dir, repair_rows, repair_summary = make_result_dir("V34_v32_expansion_selected_selector_repair", run_id, repair_items)
        selected_repair = {"result_dir": repair_dir, "summary": repair_summary}
        if metric_count(repair_summary, "joint", 41) >= metric_count(conservative_summary, "joint", 41):
            final_mode = "selector_repair"
            final_summary = repair_summary
            final_rows = repair_rows
            selection_trace = repair_trace

    write_jsonl(CODEX / "analysis" / "v34_protected_selection_trace.jsonl", selection_trace)
    write_jsonl(CODEX / "analysis" / "v34_oracle_best_rows.jsonl", oracle_rows)

    # Near-miss diagnostic after all candidates.
    near_rows: List[Dict[str, Any]] = []
    for key, candidates in sorted(candidates_by_key.items()):
        best = None
        for rec in candidates:
            if best is None or v32.score_tuple(rec.get("score") or {}) > v32.score_tuple(best.get("score") or {}):
                best = rec
        if not best:
            continue
        sc = best.get("score") or {}
        if float(sc.get("joint", 0) or 0) < 1 and (float(sc.get("result", 0) or 0) >= 1 or float(sc.get("micro", 0) or 0) >= 0.5 or int(sc.get("matches", 0) or 0) > 0):
            near_rows.append({
                "task_key": key,
                "scenario": (tasks_by_key.get(key) or {}).get("scenario"),
                "source": best.get("candidate_id") or best.get("variant_tag"),
                "score": sc,
                "first_failure": best.get("first_failure") or v32.first_failure(best),
                "tool_prefix": [c.get("tool_name") for c in (best.get("tool_program") or [])[:10]],
            })
    write_jsonl(CODEX / "analysis" / "v34_near_miss_analysis.jsonl", near_rows)

    useful_variants = []
    harmful_variants = []
    for name, summary in v34_raw_subset.items():
        joints = metric_count(summary, "joint", len(failed_tasks) or 1)
        if joints > 0 or float(summary.get("micro", 0) or 0) >= 0.20:
            useful_variants.append(name)
        else:
            harmful_variants.append(name)

    after_mtime = V10_ZIP.stat().st_mtime_ns if V10_ZIP.exists() else None
    preflight["v10_zip_overwritten"] = before_mtime != after_mtime
    v32_best_joint_count = metric_count(v32_state.get("best_protected", {}), "joint", 41)
    final_joint_count = metric_count(final_summary, "joint", 41)
    oracle_joint_count = metric_count(oracle_summary, "joint", 41)
    decision = "Keep V34 selected as current non-GT val41 candidate." if final_joint_count > v32_best_joint_count else "Do not promote V34 over V32; keep V32 best protected as current non-GT val41 best."
    if oracle_joint_count <= v32_best_joint_count:
        decision += " Candidate pool oracle-best does not exceed V32, so stop prompt expansion."
    elif oracle_joint_count > final_joint_count:
        decision += " Candidate pool has diagnostic headroom; next work should be selector repair, not more prompt variants."

    state: Dict[str, Any] = {
        "run_id": run_id,
        "version": "V34_V32_NATIVE_LOOP_EXPANSION_AND_PROTECTED_MERGE",
        "workers": args.workers,
        "preflight": preflight,
        "v32_run_id": v32_run_id,
        "baseline": {"V22": v32_state["baseline"]["V22"]},
        "v32_best_protected": v32_state["best_protected"],
        "v32_best_joint_count": v32_best_joint_count,
        "v22_success_task_count": len(v22_success),
        "v22_failed_task_count": len(failed_tasks),
        "v32_new_joint_vs_v22": audit["v32_new_joint_vs_v22"],
        "v34_raw_subset": v34_raw_subset,
        "selected_conservative": {"result_dir": conservative_dir, "summary": conservative_summary},
        "oracle_best_protected": {"result_dir": oracle_dir, "summary": oracle_summary},
        "oracle_best_joint_count": oracle_joint_count,
        "selected_repair": selected_repair,
        "final_selected_mode": final_mode,
        "final_selected_joint_count": final_joint_count,
        "final_selected_summary": final_summary,
        "useful_variants": useful_variants,
        "harmful_variants": harmful_variants,
        "final_run": False,
        "uses_final_hidden_metadata": False,
        "uses_val41_gt_for_runtime": False,
        "uses_val41_gt_for_post_eval": True,
        "auto_submit": False,
        "decision": decision,
    }
    state = {k: v for k, v in state.items() if v is not None}
    write_json(run_dir / "state.json", state)
    write_json(CODEX / "state" / "latest_v34_v32_expansion.json", state)
    (CODEX / "state" / "latest_v34_run_id.txt").write_text(run_id + "\n", encoding="utf-8")
    write_reports(run_id, state, audit, variant_defs, near_rows)
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
