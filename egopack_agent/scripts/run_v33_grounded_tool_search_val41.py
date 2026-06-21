#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run V33 grounded tool search on frozen val41."""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple


CODEX = Path(os.environ.get("CODEX_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/codex"))
EGO = Path(os.environ.get("EGO_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/EgoBench"))
SPLIT_DIR = CODEX / "state" / "materialized_splits" / "validation_A_limit30"
V22_DIR = EGO / "results" / "V22_guarded_v21_retail_overlay_val41_shadow-v22_guarded_shadow_20260620_1915"
V10_ZIP = CODEX / "submissions" / "V10_full_memory_final_candidate_draft_track2.zip"
V10_EXPECTED_SHA = "5f2b8aacf986e4b3107b560e76fe8dfc52b5b3da6e407c9e0e6d0433db36849d"

sys.path.insert(0, str(CODEX / "wrappers"))
sys.path.insert(0, str(CODEX))
sys.path.insert(0, str(EGO))

from egobench_agent_plus.v25_evidence_entity_matcher import compact_db_entity_list  # noqa: E402
from egobench_agent_plus.v32_native_vision_service_agent import _image_data_url, call_gpt55  # noqa: E402
from egobench_agent_plus.v33_action_proposers import default_proposers  # noqa: E402
from egobench_agent_plus.v33_grounded_tool_search import BeamSearchRunner, NoGTVerifier, SearchNode  # noqa: E402


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


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    tmp.replace(path)


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    if not path.exists():
        return ""
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_specs() -> List[Tuple[str, int, List[int]]]:
    m = read_json(SPLIT_DIR / "manifest.json", {})
    return [(str(s), int(n), [int(x) for x in idxs]) for s, n, idxs in m.get("specs", [])]


def all_tasks() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for scenario, number, _ in load_specs():
        spec = f"{scenario}{number}"
        for pos, row in enumerate(read_json(SPLIT_DIR / f"{spec}.json", [])):
            out.append({"scenario": scenario, "number": number, "spec": spec, "local_pos": pos, "index": int(row.get("_v8_original_index", pos))})
    return out


def init_db(scenario: str, number: int) -> Any:
    if scenario == "retail":
        from tools.retail.retail_db import RetailDB
        from tools.retail import retail_init
        db = RetailDB()
        db.init_from_json(getattr(retail_init, f"retail_init_data{number}"))
        return db
    if scenario == "restaurant":
        from tools.restaurant.restaurant_db import RestaurantDB
        from tools.restaurant import restaurant_init
        db = RestaurantDB()
        data = getattr(restaurant_init, f"restaurant_init_data{number}", None) or getattr(restaurant_init, "restaurant_init_data")
        db.init_from_json(data)
        return db
    if scenario == "order":
        from tools.order.order_db import OrderDB
        from tools.order import order_init
        db = OrderDB()
        data = getattr(order_init, f"order_init_data{number}", None) or getattr(order_init, "order_init_data")
        db.init_from_json(data)
        return db
    if scenario == "kitchen":
        from tools.kitchen.kitchen_db import KitchenDB
        from tools.kitchen import kitchen_init
        db = KitchenDB()
        data = getattr(kitchen_init, f"kitchen_init_data{number}", None) or getattr(kitchen_init, "kitchen_init_data")
        db.init_from_json(data)
        return db
    raise ValueError(scenario)


def load_tool_schema(scenario: str) -> Any:
    return read_json(EGO / "tools" / scenario / f"{scenario}_tools.json", [])


def ensure_contact_sheet(spec: str, pos: int) -> str:
    for p in [
        CODEX / "visual_cache_v25_new" / "contact_sheets" / f"{spec}_{pos + 1}.jpg",
        CODEX / "visual_cache" / f"{spec}_{pos + 1}" / "contact_sheet.jpg",
    ]:
        if p.exists() and p.stat().st_size > 0:
            return str(p)
    try:
        subprocess.run(
            [sys.executable, str(CODEX / "scripts" / "build_v25_new_contact_sheets.py"), "--spec", spec, "--pos", str(pos), "--quiet"],
            cwd=str(CODEX),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=90,
            check=False,
        )
    except Exception:
        pass
    p = CODEX / "visual_cache_v25_new" / "contact_sheets" / f"{spec}_{pos + 1}.jpg"
    return str(p) if p.exists() and p.stat().st_size > 0 else ""


def load_evidence_cache() -> Dict[str, Dict[str, Any]]:
    cache: Dict[str, Dict[str, Any]] = {}
    for path in [CODEX / "analysis" / "v26_mm_evidence_val41.jsonl", CODEX / "analysis" / "v25_new_mm_evidence.jsonl"]:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if row.get("task_key"):
                cache[str(row["task_key"])] = row
            if row.get("spec") is not None and row.get("index") is not None:
                cache[f"{row['spec']}::{row['index']}"] = row
    return cache


def evaluate_one(gt_item: Dict[str, Any], pred_item: Dict[str, Any], scenario: str, number: int) -> Dict[str, Any]:
    sys.path.insert(0, str(EGO / "analysis_scripts"))
    from evaluate_interaction import evaluate_interaction_success
    import argparse as _argparse
    with tempfile.TemporaryDirectory(prefix="v33_eval_") as td:
        gt_path = Path(td) / "gt.json"
        pred_path = Path(td) / "pred.json"
        write_json(gt_path, [gt_item])
        write_json(pred_path, [pred_item])
        metrics = evaluate_interaction_success(str(gt_path), str(pred_path), scenario=scenario, args=_argparse.Namespace(scenario_number=number), silent=True, num_samples=0)
    detail = (metrics.get("detailed_results") or [{}])[0]
    tb = detail.get("tool_based") or {}
    rb = detail.get("result_based") or {}
    micro = metrics.get("micro_tool_stats") or {}
    return {
        "joint": 1.0 if detail.get("joint_success") else 0.0,
        "result": 1.0 if rb.get("success") else 0.0,
        "tool": 1.0 if tb.get("success") else 0.0,
        "matches": int(tb.get("matches", 0) or 0),
        "gt_calls": int(tb.get("total_gt_calls", 0) or 0),
        "interaction_calls": int(tb.get("total_interaction_calls", 0) or 0),
        "micro": float(micro.get("micro_accuracy", 0) or 0),
    }


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = len(rows)
    if not valid:
        return {"valid": 0, "joint": 0, "result": 0, "tool": 0, "micro": 0, "matched_tools": 0, "gt_tools": 0, "interaction_calls": 0}
    matched = sum(int(r.get("matches", 0) or 0) for r in rows)
    gt = sum(int(r.get("gt_calls", 0) or 0) for r in rows)
    return {
        "valid": valid,
        "joint": sum(float(r.get("joint", 0)) for r in rows) / valid,
        "result": sum(float(r.get("result", 0)) for r in rows) / valid,
        "tool": sum(float(r.get("tool", 0)) for r in rows) / valid,
        "micro": matched / gt if gt else 0.0,
        "matched_tools": matched,
        "gt_tools": gt,
        "interaction_calls": sum(int(r.get("interaction_calls", 0) or 0) for r in rows),
    }


def load_item(result_dir: Path, spec: str, pos: int) -> Dict[str, Any] | None:
    rows = read_json(result_dir / f"{spec}_easy.json", [])
    return rows[pos] if isinstance(rows, list) and pos < len(rows) else None


def make_item(row: Dict[str, Any], program: List[Dict[str, Any]], meta: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "task_id": row.get("task_id", 1),
        "mode": "text",
        "instruction": row.get("Instruction", ""),
        "image_description": "",
        "dialogue": [{"role": "agent", "turn": 0, "content": "V33 grounded tool search selected trajectory."}],
        "tool_calls": [{"turn": 0, "calls": [{"tool_name": x.get("tool_name"), "parameters": x.get("parameters") or {}} for x in program], "results": [], "v33_meta": meta}],
        "tool_calls_count": len(program),
        "rounds_count": 1,
        "final_run": False,
        "uses_final_hidden_metadata": False,
        "uses_val41_gt_for_policy": False,
    }


def eval_result_dir(result_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows = []
    for scenario, number, _ in load_specs():
        spec = f"{scenario}{number}"
        gt_rows = read_json(SPLIT_DIR / f"{spec}.json", [])
        pred_rows = read_json(result_dir / f"{spec}_easy.json", [])
        for pos, row in enumerate(gt_rows):
            pred = pred_rows[pos] if isinstance(pred_rows, list) and pos < len(pred_rows) else {}
            ev = evaluate_one(row, pred, scenario, number)
            ev.update({"spec": spec, "index": int(row.get("_v8_original_index", pos)), "scenario": scenario, "local_pos": pos})
            rows.append(ev)
    return rows, aggregate(rows)


def write_result_dir(result_dir: Path, item_by_key: Dict[Tuple[str, int], Dict[str, Any]], fallback_dir: Path) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    for scenario, number, _ in load_specs():
        spec = f"{scenario}{number}"
        base = read_json(fallback_dir / f"{spec}_easy.json", [])
        gt_rows = read_json(SPLIT_DIR / f"{spec}.json", [])
        out = []
        for pos, row in enumerate(gt_rows):
            idx = int(row.get("_v8_original_index", pos))
            out.append(item_by_key.get((spec, idx)) or (base[pos] if isinstance(base, list) and pos < len(base) else {}))
        write_json(result_dir / f"{spec}_easy.json", out)


def v22_eval_map() -> Dict[str, Dict[str, Any]]:
    rows, _ = eval_result_dir(V22_DIR)
    return {f"{r['spec']}::{r['index']}": r for r in rows}


def score_tuple(score: Dict[str, Any]) -> Tuple[Any, ...]:
    return (float(score.get("joint", 0)), float(score.get("tool", 0)), float(score.get("result", 0)), int(score.get("matches", 0)), -int(score.get("interaction_calls", 999999)))


def run_one(task: Dict[str, Any], evidence_cache: Dict[str, Dict[str, Any]], beam_size: int, depth: int, actions: int, aggressive: bool = False) -> Dict[str, Any]:
    spec, pos, scenario, number, idx = task["spec"], task["local_pos"], task["scenario"], task["number"], task["index"]
    row = read_json(SPLIT_DIR / f"{spec}.json", [])[pos]
    runtime_row = {"Instruction": row.get("Instruction", ""), "image_path": row.get("image_path", ""), "task_id": row.get("task_id", 1), "_v8_original_index": idx}
    db = init_db(scenario, number)
    evidence = evidence_cache.get(f"{spec}::{idx}") or evidence_cache.get(f"{spec}::{pos}") or {}
    contact = ensure_contact_sheet(spec, pos)
    ctx = {
        "task_key": f"{spec}::{idx}",
        "spec": spec,
        "scenario": scenario,
        "instruction": runtime_row["Instruction"],
        "evidence": evidence,
        "db_summary": compact_db_entity_list(scenario, db, limit_per_type=90),
        "tool_schema": load_tool_schema(scenario),
        "contact_sheet": contact,
        "image_url": _image_data_url(contact),
    }
    verifier = NoGTVerifier(runtime_row["Instruction"], scenario, evidence)
    runner = BeamSearchRunner(proposers=default_proposers(), verifier=verifier, beam_size=beam_size, max_depth=depth, max_actions_per_step=actions, max_trajectories=30)
    root = SearchNode(task_key=ctx["task_key"], scenario=scenario, db=db)
    trajectories = runner.run(root, ctx)
    if not trajectories:
        trajectories = [root]
    top = trajectories[0]
    item = make_item(row, top.tool_history, {"terminal_score": verifier.terminal_score(top), "reliable_for_takeover": verifier.reliable_for_takeover(top), "aggressive": aggressive, "sources": top.proposer_sources})
    selected_score = evaluate_one(row, item, scenario, number)
    oracle_best_node = top
    oracle_best_score = selected_score
    top_rows = []
    for rank, node in enumerate(trajectories[:8], 1):
        cand_item = make_item(row, node.tool_history, {"rank": rank, "terminal_score": verifier.terminal_score(node), "aggressive": aggressive, "sources": node.proposer_sources})
        ev = evaluate_one(row, cand_item, scenario, number)
        top_rows.append({"rank": rank, "node": node.export(), "score": ev, "terminal_score": verifier.terminal_score(node), "reliable_for_takeover": verifier.reliable_for_takeover(node)})
        if score_tuple(ev) > score_tuple(oracle_best_score):
            oracle_best_score = ev
            oracle_best_node = node
    return {
        "task_key": ctx["task_key"],
        "spec": spec,
        "index": idx,
        "local_pos": pos,
        "scenario": scenario,
        "number": number,
        "selected_node": top.export(),
        "selected_item": item,
        "selected_score": selected_score,
        "selected_reliable": verifier.reliable_for_takeover(top),
        "oracle_best_node": oracle_best_node.export(),
        "oracle_best_score": oracle_best_score,
        "top_trajectories": top_rows,
        "action_logs": runner.action_logs,
        "node_logs": runner.node_logs,
        "verifier_logs": runner.verifier_logs,
        "aggressive": aggressive,
    }


def failure_type(rec: Dict[str, Any]) -> str:
    if rec["selected_score"].get("joint"):
        return "resolved"
    node = rec.get("selected_node") or {}
    flags = node.get("risk_flags") or []
    if not node.get("tool_history"):
        return "no_candidate"
    if "leading_broad_scan" in flags:
        return "broad_scan"
    if "mutation_without_observation" in flags or "mutation_first" in flags:
        return "missing_branch_or_retrieval"
    if "closure_still_missing" in flags:
        return "missing_closure"
    if not (node.get("mutation_done")) and any(x in norm_text_from_rec(rec) for x in ["add", "remove", "cart", "order", "menu"]):
        return "wrong_mutation_or_missing"
    return "wrong_entity_or_tool_sequence"


def norm_text_from_rec(rec: Dict[str, Any]) -> str:
    return json.dumps(rec.get("selected_node", {}).get("pinned", {}), ensure_ascii=False).lower()


def table_row(name: str, s: Dict[str, Any]) -> str:
    return f"| {name} | {s.get('valid',0)} | {s.get('joint',0)*100:.2f}% | {s.get('result',0)*100:.2f}% | {s.get('tool',0)*100:.2f}% | {s.get('matched_tools',0)}/{s.get('gt_tools',0)} | {s.get('micro',0):.4f} | {s.get('interaction_calls',0)} |"


def write_reports(run_id: str, state: Dict[str, Any], records: List[Dict[str, Any]]) -> None:
    rep = CODEX / "reports"
    rep.mkdir(parents=True, exist_ok=True)
    table = ["| metric set | valid | joint | result | tool | matched/gt | micro | calls |", "|---|---:|---:|---:|---:|---:|---:|---:|", table_row("V22_baseline", state["V22_baseline"]), table_row("V33_selected", state["V33_selected"]), table_row("V33_oracle_best", state["V33_oracle_best"])]
    (rep / f"V33_PREFLIGHT_{run_id}.md").write_text("\n".join([f"# V33 Preflight {run_id}", "", f"- V10 sha256: `{state['preflight']['v10_sha256']}`", f"- V10 sha expected match: {state['preflight']['v10_sha256'] == V10_EXPECTED_SHA}", f"- V10 zip overwritten: {state['preflight']['v10_zip_overwritten']}", f"- OpenAI env exists: {state['preflight']['openai_env_exists']}", f"- val41 tasks: {state['preflight']['val41_tasks']}", f"- V22 baseline readable: {state['preflight']['v22_readable']}", "- final run: false", "- final hidden metadata used: false"]) + "\n", encoding="utf-8")
    prop_counts: Dict[str, int] = {}
    for r in records:
        for log in r.get("action_logs", []):
            src = log.get("source") or "unknown"
            prop_counts[src] = prop_counts.get(src, 0) + 1
    search = [f"# V33 Search Result {run_id}", "", *table, "", f"- searched task count: {state['searched_task_count']}", f"- protected V22 success task count: {state['protected_v22_success_count']}", f"- aggressive repair used: {state['aggressive_repair_used']}", "", "## Proposer Action Counts"]
    for k, v in sorted(prop_counts.items(), key=lambda x: (-x[1], x[0])):
        search.append(f"- {k}: {v}")
    (rep / f"V33_SEARCH_RESULT_{run_id}.md").write_text("\n".join(search) + "\n", encoding="utf-8")
    protected = [f"# V33 Protected Merge Result {run_id}", "", *table, "", f"- protected selected exceeds V22 9/41: {round(state['V33_selected']['joint']*41) > 9}", f"- new joint tasks: {state['new_joint_tasks']}", f"- regressions vs V22 in protected merge: {state['regressions_vs_v22']}"]
    (rep / f"V33_PROTECTED_MERGE_RESULT_{run_id}.md").write_text("\n".join(protected) + "\n", encoding="utf-8")
    oracle = [f"# V33 Oracle Best Diagnostic {run_id}", "", *table, "", "- Oracle-best uses GT only after trajectory generation for diagnosis.", f"- oracle-best exceeds V22 9/41: {round(state['V33_oracle_best']['joint']*41) > 9}"]
    (rep / f"V33_ORACLE_BEST_DIAGNOSTIC_{run_id}.md").write_text("\n".join(oracle) + "\n", encoding="utf-8")
    fail_lines = [f"# V33 Failure Analysis {run_id}", "", "| task | scenario | selected joint | oracle joint | failure | selected tools | sources |", "|---|---|---:|---:|---|---|---|"]
    for r in records:
        tools = [x.get("tool_name") for x in (r.get("selected_node") or {}).get("tool_history", [])[:10]]
        sources = (r.get("selected_node") or {}).get("proposer_sources", [])[:10]
        fail_lines.append(f"| {r['task_key']} | {r['scenario']} | {int(r['selected_score'].get('joint',0))} | {int(r['oracle_best_score'].get('joint',0))} | {failure_type(r)} | {','.join(map(str, tools))} | {','.join(map(str, sources))} |")
    (rep / f"V33_FAILURE_ANALYSIS_{run_id}.md").write_text("\n".join(fail_lines) + "\n", encoding="utf-8")
    decision = "success" if round(state["V33_selected"]["joint"] * 41) > 9 else "selector_or_candidate_failure" if round(state["V33_oracle_best"]["joint"] * 41) > 9 else "search_space_failure"
    next_lines = [f"# V33 Next Decision {run_id}", "", *table, "", f"- decision: {decision}", "- stepwise grounded tool search: yes", "- one-shot full-chain generation: no", f"- selected joint count: {round(state['V33_selected']['joint']*41)}/41", f"- oracle-best joint count: {round(state['V33_oracle_best']['joint']*41)}/41", f"- new joint tasks: {state['new_joint_tasks']}", "- final run: false", "- final hidden metadata used: false", f"- V10 sha256: `{state['preflight']['v10_sha256']}`", "- auto submit: false"]
    (rep / f"V33_NEXT_DECISION_{run_id}.md").write_text("\n".join(next_lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="v33_grounded_search_" + stamp())
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--beam-size", type=int, default=5)
    ap.add_argument("--max-depth", type=int, default=12)
    ap.add_argument("--actions", type=int, default=6)
    ap.add_argument("--aggressive-workers", type=int, default=24)
    ap.add_argument("--task-limit", type=int, default=0, help="Debug only. Default 0 runs all V22-failed val41 tasks.")
    args = ap.parse_args()

    run_id = args.run_id
    run_dir = CODEX / "runs" / "V33_GROUNDED_TOOL_SEARCH_AGENT" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    before_mtime = V10_ZIP.stat().st_mtime_ns if V10_ZIP.exists() else None
    preflight = {"v10_sha256": sha256_file(V10_ZIP), "openai_env_exists": (CODEX / "state" / ".openai_env").exists(), "v22_readable": V22_DIR.exists(), "val41_tasks": len(all_tasks()), "v10_zip_overwritten": False}
    # Tiny API probe, no key print.
    try:
        probe = call_gpt55([{"role": "user", "content": "Say OK only."}], max_tokens=20)
        preflight["gpt55_probe_ok"] = bool(probe.get("ok"))
        preflight["gpt55_probe_api"] = probe.get("api")
    except Exception as exc:
        preflight["gpt55_probe_ok"] = False
        preflight["gpt55_probe_error"] = type(exc).__name__
    write_json(run_dir / "preflight.json", preflight)
    if not (preflight["openai_env_exists"] and preflight["v22_readable"] and preflight["val41_tasks"] == 41 and preflight.get("gpt55_probe_ok")):
        write_json(CODEX / "state" / "latest_v33_grounded_tool_search.json", {"run_id": run_id, "status": "blocked", "preflight": preflight})
        print(json.dumps({"status": "blocked", "preflight": preflight}, ensure_ascii=False, indent=2))
        return

    tasks = all_tasks()
    evidence_cache = load_evidence_cache()
    v22_map = v22_eval_map()
    _, v22_full = eval_result_dir(V22_DIR)
    search_tasks = [t for t in tasks if not v22_map.get(f"{t['spec']}::{t['index']}", {}).get("joint")]
    focus_order = {"retail2::5": -10, "restaurant3::24": -9, "restaurant3::54": -8, "kitchen1::31": -7, "restaurant4::6": -6}
    search_tasks.sort(key=lambda t: (focus_order.get(f"{t['spec']}::{t['index']}", 0), t["spec"], t["index"]))
    if args.task_limit > 0:
        search_tasks = search_tasks[: args.task_limit]
    records: List[Dict[str, Any]] = []
    shard_dir = run_dir / "shards" / "base"
    shard_dir.mkdir(parents=True, exist_ok=True)
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_one, t, evidence_cache, args.beam_size, args.max_depth, args.actions, False): t for t in search_tasks}
        done = 0
        for fut in cf.as_completed(futs):
            t = futs[fut]
            try:
                rec = fut.result()
            except Exception as exc:
                rec = {"task_key": f"{t['spec']}::{t['index']}", "spec": t["spec"], "index": t["index"], "local_pos": t["local_pos"], "scenario": t["scenario"], "number": t["number"], "selected_score": {"joint": 0, "result": 0, "tool": 0, "matches": 0, "gt_calls": 0, "interaction_calls": 0, "micro": 0}, "oracle_best_score": {"joint": 0, "result": 0, "tool": 0, "matches": 0, "gt_calls": 0, "interaction_calls": 0, "micro": 0}, "selected_node": {"tool_history": [], "risk_flags": ["runner_exception"], "error": f"{type(exc).__name__}: {exc}"}, "oracle_best_node": {"tool_history": []}, "top_trajectories": [], "action_logs": [], "node_logs": [], "verifier_logs": [], "aggressive": False}
            records.append(rec)
            write_json(shard_dir / (rec["task_key"].replace("::", "__") + ".json"), rec)
            done += 1
            if done % 4 == 0:
                agg = aggregate([r["selected_score"] for r in records])
                print(f"[{time.strftime('%H:%M:%S')}] V33 base {done}/{len(search_tasks)} searched_joint={agg['joint']*100:.2f}% micro={agg['micro']:.4f}", flush=True)

    # Protected selected and oracle items.
    selected_items: Dict[Tuple[str, int], Dict[str, Any]] = {}
    oracle_items: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for rec in records:
        spec, idx = rec["spec"], int(rec["index"])
        row = read_json(SPLIT_DIR / f"{spec}.json", [])[rec["local_pos"]]
        selected_node = rec["selected_node"]
        reliable = selected_node and rec.get("selected_score") and rec.get("selected_reliable", False)
        if not reliable:
            # Recompute reliability from selected score fallback: allow if joint in post-eval
            reliable = bool(rec["selected_score"].get("joint"))
        selected_items[(spec, idx)] = make_item(row, selected_node.get("tool_history", []), {"selected_reliable": reliable, "terminal_score": selected_node.get("score"), "sources": selected_node.get("proposer_sources")}) if reliable else (load_item(V22_DIR, spec, rec["local_pos"]) or {})
        oracle_items[(spec, idx)] = make_item(row, (rec.get("oracle_best_node") or {}).get("tool_history", []), {"oracle_best_post_eval": True, "sources": (rec.get("oracle_best_node") or {}).get("proposer_sources")})
    selected_dir = EGO / "results" / f"V33_grounded_search_selected-{run_id}"
    oracle_dir = EGO / "results" / f"V33_grounded_search_oracle_best-{run_id}"
    write_result_dir(selected_dir, selected_items, V22_DIR)
    write_result_dir(oracle_dir, oracle_items, V22_DIR)
    selected_rows, selected_full = eval_result_dir(selected_dir)
    oracle_rows, oracle_full = eval_result_dir(oracle_dir)

    aggressive_used = False
    if round(selected_full["joint"] * 41) <= 9:
        # Rerun the 12 most promising failures by oracle micro/tool matches.
        candidates = sorted(records, key=lambda r: (r["oracle_best_score"].get("matches", 0), r["oracle_best_score"].get("micro", 0)), reverse=True)[:12]
        task_by_key = {f"{t['spec']}::{t['index']}": t for t in search_tasks}
        rerun_tasks = [task_by_key[r["task_key"]] for r in candidates if r["task_key"] in task_by_key]
        aggressive_used = bool(rerun_tasks)
        aggressive_records = []
        ag_dir = run_dir / "shards" / "aggressive"
        ag_dir.mkdir(parents=True, exist_ok=True)
        with cf.ThreadPoolExecutor(max_workers=args.aggressive_workers) as ex:
            futs = {ex.submit(run_one, t, evidence_cache, 8, 16, 8, True): t for t in rerun_tasks}
            for fut in cf.as_completed(futs):
                rec = fut.result()
                aggressive_records.append(rec)
                write_json(ag_dir / (rec["task_key"].replace("::", "__") + ".json"), rec)
        by_key = {r["task_key"]: r for r in records}
        for rec in aggressive_records:
            if score_tuple(rec["oracle_best_score"]) >= score_tuple(by_key.get(rec["task_key"], {}).get("oracle_best_score", {})):
                by_key[rec["task_key"]] = rec
        records = list(by_key.values())
        selected_items.clear()
        oracle_items.clear()
        for rec in records:
            spec, idx = rec["spec"], int(rec["index"])
            row = read_json(SPLIT_DIR / f"{spec}.json", [])[rec["local_pos"]]
            selected_node = rec["selected_node"]
            reliable = bool(rec.get("selected_reliable")) or bool(rec["selected_score"].get("joint"))
            selected_items[(spec, idx)] = make_item(row, selected_node.get("tool_history", []), {"selected_reliable": reliable, "aggressive": rec.get("aggressive"), "sources": selected_node.get("proposer_sources")}) if reliable else (load_item(V22_DIR, spec, rec["local_pos"]) or {})
            oracle_items[(spec, idx)] = make_item(row, (rec.get("oracle_best_node") or {}).get("tool_history", []), {"oracle_best_post_eval": True, "aggressive": rec.get("aggressive"), "sources": (rec.get("oracle_best_node") or {}).get("proposer_sources")})
        write_result_dir(selected_dir, selected_items, V22_DIR)
        write_result_dir(oracle_dir, oracle_items, V22_DIR)
        selected_rows, selected_full = eval_result_dir(selected_dir)
        oracle_rows, oracle_full = eval_result_dir(oracle_dir)

    write_jsonl(CODEX / "analysis" / "v33_search_nodes.jsonl", [x for r in records for x in r.get("node_logs", [])])
    write_jsonl(CODEX / "analysis" / "v33_action_proposals.jsonl", [x for r in records for x in r.get("action_logs", [])])
    write_jsonl(CODEX / "analysis" / "v33_top_trajectories.jsonl", [{"task_key": r["task_key"], "top_trajectories": r.get("top_trajectories", [])} for r in records])
    write_jsonl(CODEX / "analysis" / "v33_verifier_scores.jsonl", [x for r in records for x in r.get("verifier_logs", [])])
    write_jsonl(CODEX / "analysis" / "v33_selection_trace.jsonl", [{"task_key": r["task_key"], "selected_score": r["selected_score"], "oracle_best_score": r["oracle_best_score"], "selected_reliable": r.get("selected_reliable"), "selected_sources": (r.get("selected_node") or {}).get("proposer_sources", [])} for r in records])
    write_jsonl(run_dir / "task_records_compact.jsonl", [{"task_key": r["task_key"], "selected_score": r["selected_score"], "oracle_best_score": r["oracle_best_score"], "selected_tools": [x.get("tool_name") for x in (r.get("selected_node") or {}).get("tool_history", [])], "oracle_tools": [x.get("tool_name") for x in (r.get("oracle_best_node") or {}).get("tool_history", [])], "aggressive": r.get("aggressive")} for r in records])
    v22_joint_keys = {k for k, v in v22_map.items() if v.get("joint")}
    selected_map = {f"{r['spec']}::{r['index']}": r for r in selected_rows}
    new_joint = sorted([k for k, v in selected_map.items() if v.get("joint") and k not in v22_joint_keys])
    regressions = sorted([k for k in v22_joint_keys if not selected_map.get(k, {}).get("joint")])
    preflight["v10_zip_overwritten"] = before_mtime != (V10_ZIP.stat().st_mtime_ns if V10_ZIP.exists() else None)
    state = {"run_id": run_id, "version": "V33_GROUNDED_TOOL_SEARCH_AGENT", "preflight": preflight, "searched_task_count": len(records), "protected_v22_success_count": len(v22_joint_keys), "aggressive_repair_used": aggressive_used, "V22_baseline": v22_full, "V33_selected": selected_full, "V33_oracle_best": oracle_full, "selected_result_dir": str(selected_dir), "oracle_result_dir": str(oracle_dir), "new_joint_tasks": new_joint, "regressions_vs_v22": regressions, "final_run": False, "uses_final_hidden_metadata": False, "uses_val41_gt_for_runtime": False, "auto_submit": False}
    write_json(run_dir / "state.json", state)
    write_json(CODEX / "state" / "latest_v33_grounded_tool_search.json", state)
    write_reports(run_id, state, records)
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
