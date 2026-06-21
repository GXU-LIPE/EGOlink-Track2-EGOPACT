#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V37 smoke and mini-shadow for V34 native-loop chain after official sync."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


CODEX = Path(os.environ.get("CODEX_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/codex"))
EGO = Path(os.environ.get("EGO_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/EgoBench"))
V10_ZIP = CODEX / "submissions" / "V10_full_memory_final_candidate_draft_track2.zip"
V10_EXPECTED_SHA = "5f2b8aacf986e4b3107b560e76fe8dfc52b5b3da6e407c9e0e6d0433db36849d"

sys.path.insert(0, str(CODEX / "scripts"))
sys.path.insert(0, str(CODEX / "wrappers"))
sys.path.insert(0, str(CODEX))
sys.path.insert(0, str(EGO))

from egobench_agent_plus import v37_official_adapter as adapter  # noqa: E402

v32 = importlib.import_module("run_v32_native_vision_val41")
service_agent = importlib.import_module("egobench_agent_plus.v32_native_vision_service_agent")


BASE_OUTPUT_RULES = """Output rules:
- If tools are needed, output exactly one JSON array and nothing else.
- Each array item must be {"tool_name": "...", "parameters": {...}}.
- If no more tools are needed, output a short final response in plain text.
- Never mix prose and JSON in one response.
- Use only tools listed in current official schema.
- Use current task, DB context, visual/contact-sheet evidence, and tool observations only.
- Do not use final hidden metadata or val41 GT for runtime policy.
"""


V37_PROMPTS = {
    "v37_compact_official": f"""You are the EgoBench Track2 service agent running in the official tool loop.
Complete the current user task with the shortest correct official tool process.
{BASE_OUTPUT_RULES}
""",
    "v37_closure_guarded": f"""You are the EgoBench Track2 service agent. Resolve canonical entities, call narrow tools, mutate if requested, then compute only the requested closure.
If the user asks payment, tax, nutrition, order/cart/list summary, compute it near the end.
{BASE_OUTPUT_RULES}
""",
}


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
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
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
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def patch_v37_prompt_router() -> None:
    def variant_prompt(name: str, repair_hint: str = "") -> str:
        prompt = V37_PROMPTS.get(name) or V37_PROMPTS["v37_compact_official"]
        if repair_hint:
            prompt += "\nRuntime repair hint:\n" + repair_hint[:1200] + "\n"
        return prompt

    service_agent.variant_prompt = variant_prompt

    def compact_tool_schema(tool_schema: Any) -> Any:
        if not isinstance(tool_schema, list):
            return tool_schema
        out: List[Dict[str, Any]] = []
        for tool in tool_schema[:90]:
            if not isinstance(tool, dict):
                continue
            fn = tool.get("function")
            if isinstance(fn, dict):
                name = fn.get("name") or fn.get("tool_name")
                desc = fn.get("description", "")
                params = fn.get("parameters")
            else:
                name = tool.get("name") or tool.get("tool_name")
                desc = tool.get("description", "")
                params = tool.get("parameters") or tool.get("input_schema")
            if not name:
                continue
            row = {"name": str(name), "description": str(desc)[:500]}
            if isinstance(params, dict):
                row["parameters"] = params
            out.append(row)
        return out

    service_agent._compact_tool_schema = compact_tool_schema


def safe_make_item(row: Dict[str, Any], agent_item: Dict[str, Any], variant: str) -> Dict[str, Any]:
    item = copy.deepcopy(agent_item)
    item["task_id"] = row.get("task_id", 1)
    item["instruction"] = row.get("Instruction", "")
    item["image_description"] = ""
    item["mode"] = "text"
    item["final_run"] = False
    item["uses_final_hidden_metadata"] = False
    item["uses_val41_gt_for_policy"] = False
    item.setdefault("v37_meta", {})["variant"] = variant
    return item


def evaluate_one(gt_item: Dict[str, Any], pred_item: Dict[str, Any], scenario: str, number: int) -> Dict[str, Any]:
    sys.path.insert(0, str(EGO / "analysis_scripts"))
    from evaluate_interaction import evaluate_interaction_success
    import argparse as _argparse
    import tempfile

    with tempfile.TemporaryDirectory(prefix="v37_eval_") as td:
        gt_path = Path(td) / "gt.json"
        pred_path = Path(td) / "pred.json"
        write_json(gt_path, [gt_item])
        write_json(pred_path, [pred_item])
        metrics = evaluate_interaction_success(
            str(gt_path),
            str(pred_path),
            scenario=scenario,
            args=_argparse.Namespace(scenario_number=number),
            silent=True,
            num_samples=0,
        )
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
    if not rows:
        return {"valid": 0, "joint": 0, "result": 0, "tool": 0, "micro": 0, "matched_tools": 0, "gt_tools": 0, "interaction_calls": 0}
    matched = sum(int(r.get("matches", 0) or 0) for r in rows)
    gt = sum(int(r.get("gt_calls", 0) or 0) for r in rows)
    return {
        "valid": len(rows),
        "joint": sum(float(r.get("joint", 0) or 0) for r in rows) / len(rows),
        "result": sum(float(r.get("result", 0) or 0) for r in rows) / len(rows),
        "tool": sum(float(r.get("tool", 0) or 0) for r in rows) / len(rows),
        "micro": matched / gt if gt else 0.0,
        "matched_tools": matched,
        "gt_tools": gt,
        "interaction_calls": sum(int(r.get("interaction_calls", 0) or 0) for r in rows),
    }


def _video_exists(row: Dict[str, Any]) -> bool:
    val = row.get("image_path") or row.get("video_path") or ""
    if not val:
        return False
    base = Path(str(val)).name
    return (EGO / "videos" / base).exists() or Path(str(val)).exists()


def smoke_a_load(tasks: List[Dict[str, Any]], out_path: Path) -> Tuple[List[Dict[str, Any]], bool]:
    selected = adapter.select_one_per_scenario(tasks)
    rows: List[Dict[str, Any]] = []
    ok = True
    for t in selected:
        try:
            db = adapter.init_db(t["scenario"], t["number"])
            tools = adapter.load_tool_schema(t["scenario"])
            db_info = adapter.db_entity_counts_and_samples(t["scenario"], db)
            row = t["row"]
            trace = {
                "task_key": f"{t['spec']}::{t['index']}",
                "scenario": t["scenario"],
                "status": "ok",
                "instruction_present": bool(row.get("Instruction")),
                "video_exists": _video_exists(row),
                "tool_count": len(tools),
                "db_class": type(db).__name__,
                "db_counts": db_info.get("counts", {}),
                "runtime_excluded_fields": ["analysis", "ground_truth", "image_description"],
            }
            if not trace["instruction_present"] or not trace["tool_count"]:
                ok = False
                trace["status"] = "error"
        except Exception as exc:
            ok = False
            trace = {"task_key": f"{t['spec']}::{t['index']}", "scenario": t["scenario"], "status": "error", "error": f"{type(exc).__name__}: {exc}"}
        rows.append(trace)
    write_jsonl(out_path, rows)
    return rows, ok


def smoke_b_dryrun(tasks: List[Dict[str, Any]], out_path: Path) -> Tuple[List[Dict[str, Any]], bool]:
    selected = adapter.select_one_per_scenario(tasks)
    rows: List[Dict[str, Any]] = []
    ok = True
    for t in selected:
        try:
            db = adapter.init_db(t["scenario"], t["number"])
            calls = adapter.generate_harmless_query_calls(t["scenario"], db, limit=2)
            results = [adapter.execute_tool_call(db, c) for c in calls]
            status = "ok" if calls and all(r.get("status") == "success" for r in results) else "error"
            if status != "ok":
                ok = False
            rows.append(
                {
                    "task_key": f"{t['spec']}::{t['index']}",
                    "scenario": t["scenario"],
                    "status": status,
                    "calls": calls,
                    "results": results,
                    "mutation_used": any(adapter.MUTATION_RE.search(c.get("tool_name", "")) for c in calls),
                }
            )
        except Exception as exc:
            ok = False
            rows.append({"task_key": f"{t['spec']}::{t['index']}", "scenario": t["scenario"], "status": "error", "error": f"{type(exc).__name__}: {exc}"})
    write_jsonl(out_path, rows)
    return rows, ok


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
            key = row.get("task_key")
            spec = row.get("spec")
            idx = row.get("index")
            if key:
                cache[str(key)] = row
            if spec is not None and idx is not None:
                cache[f"{spec}::{idx}"] = row
    return cache


def run_native_task(task: Dict[str, Any], variant: str, max_rounds: int, max_tool_calls: int, evidence_cache: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    spec, idx, pos = task["spec"], task["index"], task["local_pos"]
    scenario, number = task["scenario"], task["number"]
    row = task["row"]
    runtime_row = adapter.safe_runtime_row(row, idx)
    db = adapter.init_db(scenario, number)
    db_summary = v32.compact_db_entity_list(scenario, db, limit_per_type=90)
    tool_schema = adapter.load_tool_schema(scenario)
    contact = v32.ensure_contact_sheet(spec, pos, runtime_row)
    evidence = evidence_cache.get(f"{spec}::{idx}") or evidence_cache.get(f"{spec}::{pos}") or {}
    trace = service_agent.run_native_service_agent(
        row=runtime_row,
        scenario=scenario,
        spec=spec,
        db=db,
        db_summary=db_summary,
        tool_schema=tool_schema,
        contact_sheet_path=contact,
        evidence=evidence,
        variant=variant,
        repair_hint="",
        max_rounds=max_rounds,
        max_tool_calls=max_tool_calls,
    )
    item = safe_make_item(row, trace["item"], variant)
    score = evaluate_one(row, item, scenario, number)
    return {
        "task_key": f"{spec}::{idx}",
        "spec": spec,
        "index": idx,
        "local_pos": pos,
        "scenario": scenario,
        "number": number,
        "variant": variant,
        "score": score,
        "tool_program": trace.get("tool_program") or [],
        "risk_flags": trace.get("risk_flags") or [],
        "api_errors": trace.get("api_errors") or [],
        "vision_success": trace.get("vision_success", False),
        "contact_sheet": contact,
        "final_text": trace.get("final_text", ""),
        "item": item,
    }


def _timeout_handler(signum: int, frame: Any) -> None:
    raise TimeoutError("candidate_timeout")


def run_native_task_with_timeout(
    task: Dict[str, Any],
    variant: str,
    max_rounds: int,
    max_tool_calls: int,
    evidence_cache: Dict[str, Dict[str, Any]],
    timeout_s: int,
) -> Dict[str, Any]:
    if timeout_s <= 0:
        return run_native_task(task, variant, max_rounds, max_tool_calls, evidence_cache)
    old_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_s)
    try:
        return run_native_task(task, variant, max_rounds, max_tool_calls, evidence_cache)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def smoke_c_native(tasks: List[Dict[str, Any]], out_path: Path, max_rounds: int, max_tool_calls: int) -> Tuple[List[Dict[str, Any]], bool]:
    patch_v37_prompt_router()
    selected = adapter.select_one_per_scenario(tasks)
    evidence_cache = load_evidence_cache()
    rows: List[Dict[str, Any]] = []
    ok = True
    for t in selected:
        try:
            rec = run_native_task(t, "v37_compact_official", max_rounds=max_rounds, max_tool_calls=max_tool_calls, evidence_cache=evidence_cache)
            public = {k: v for k, v in rec.items() if k != "item"}
            rows.append(public)
            if rec.get("api_errors"):
                ok = False
        except Exception as exc:
            ok = False
            rows.append(
                {
                    "task_key": f"{t['spec']}::{t['index']}",
                    "scenario": t["scenario"],
                    "variant": "v37_compact_official",
                    "score": {"joint": 0, "result": 0, "tool": 0, "matches": 0, "gt_calls": 0, "interaction_calls": 0, "micro": 0},
                    "tool_program": [],
                    "risk_flags": ["runner_exception"],
                    "api_errors": [f"{type(exc).__name__}: {exc}"],
                    "vision_success": False,
                    "final_text": "",
                }
            )
    write_jsonl(out_path, rows)
    return rows, ok


def run_mini_shadow(tasks: List[Dict[str, Any]], out_path: Path, selection_path: Path, max_rounds: int, max_tool_calls: int, timeout_s: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    patch_v37_prompt_router()
    selected_tasks = adapter.select_mini_shadow_tasks(tasks, limit=12)
    evidence_cache = load_evidence_cache()
    rows: List[Dict[str, Any]] = []
    selection: List[Dict[str, Any]] = []
    out_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("", encoding="utf-8")
    selection_path.write_text("", encoding="utf-8")
    for t in selected_tasks:
        candidates: List[Dict[str, Any]] = []
        for variant in ["v37_compact_official", "v37_closure_guarded"]:
            try:
                candidates.append(run_native_task_with_timeout(t, variant, max_rounds=max_rounds, max_tool_calls=max_tool_calls, evidence_cache=evidence_cache, timeout_s=timeout_s))
            except Exception as exc:
                candidates.append(
                    {
                        "task_key": f"{t['spec']}::{t['index']}",
                        "spec": t["spec"],
                        "index": t["index"],
                        "local_pos": t["local_pos"],
                        "scenario": t["scenario"],
                        "number": t["number"],
                        "variant": variant,
                        "score": {"joint": 0, "result": 0, "tool": 0, "matches": 0, "gt_calls": 0, "interaction_calls": 0, "micro": 0},
                        "tool_program": [],
                        "risk_flags": ["runner_exception"],
                        "api_errors": [f"{type(exc).__name__}: {exc}"],
                        "vision_success": False,
                        "contact_sheet": "",
                        "final_text": "",
                        "item": {},
                    }
                )
        candidates.sort(
            key=lambda r: (
                float(r["score"].get("joint", 0)),
                float(r["score"].get("tool", 0)),
                float(r["score"].get("result", 0)),
                int(r["score"].get("matches", 0)),
                -int(r["score"].get("interaction_calls", 999999)),
            ),
            reverse=True,
        )
        chosen = candidates[0]
        public = {k: v for k, v in chosen.items() if k != "item"}
        rows.append(public)
        sel = {
            "task_key": chosen["task_key"],
            "selected_variant": chosen["variant"],
            "selected_score": chosen["score"],
            "candidate_scores": [{k: v for k, v in c.items() if k in {"variant", "score", "risk_flags", "api_errors", "vision_success"}} for c in candidates],
            "selection_note": "diagnostic best-of among V37 prompt variants for mini shadow only",
        }
        selection.append(sel)
        append_jsonl(out_path, public)
        append_jsonl(selection_path, sel)
        print(f"[mini] {chosen['task_key']} selected={chosen['variant']} joint={chosen['score'].get('joint')} matches={chosen['score'].get('matches')}/{chosen['score'].get('gt_calls')}", flush=True)
    return rows, aggregate([r["score"] for r in rows])


def table_row(name: str, s: Dict[str, Any]) -> str:
    return f"| {name} | {s.get('valid', 0)} | {s.get('joint', 0)*100:.2f}% | {s.get('result', 0)*100:.2f}% | {s.get('tool', 0)*100:.2f}% | {s.get('matched_tools', 0)}/{s.get('gt_tools', 0)} | {s.get('micro', 0):.4f} | {s.get('interaction_calls', 0)} |"


def write_smoke_report(run_id: str, state: Dict[str, Any], smoke_a: List[Dict[str, Any]], smoke_b: List[Dict[str, Any]], smoke_c: List[Dict[str, Any]]) -> Path:
    report = CODEX / "reports" / f"V37_V34_CHAIN_SMOKE_RESULT_{run_id}.md"
    c_summary = aggregate([r.get("score", {}) for r in smoke_c])
    lines = [
        "# V37 V34 Chain Smoke Result",
        "",
        f"- run_id: `{run_id}`",
        f"- Smoke A load passed: {state.get('smoke_a_passed')}",
        f"- Smoke B tool dry-run passed: {state.get('smoke_b_passed')}",
        f"- Smoke C V34 native loop passed: {state.get('smoke_c_passed')}",
        f"- GPT/API calls used in Smoke C: {len(smoke_c)} tasks",
        f"- final run: {state.get('final_run')}",
        f"- final hidden metadata used: {state.get('uses_final_hidden_metadata')}",
        f"- V10 sha matches expected: {state.get('v10_sha256') == V10_EXPECTED_SHA}",
        "",
        "## Smoke C Diagnostic Metrics",
        "",
        "| set | valid | joint | result | tool | matched/gt | micro | calls |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        table_row("Smoke C", c_summary),
        "",
        "## Smoke A Load",
        "",
    ]
    for row in smoke_a:
        lines.append(f"- {row.get('task_key')} {row.get('scenario')}: {row.get('status')} video={row.get('video_exists')} tools={row.get('tool_count')}")
    lines.extend(["", "## Smoke B Dry Run", ""])
    for row in smoke_b:
        calls = [c.get("tool_name") for c in row.get("calls", [])]
        lines.append(f"- {row.get('task_key')} {row.get('scenario')}: {row.get('status')} calls={calls}")
    lines.extend(["", "## Smoke C Native Loop", ""])
    for row in smoke_c:
        score = row.get("score", {})
        names = [c.get("tool_name") for c in row.get("tool_program", [])]
        lines.append(f"- {row.get('task_key')} {row.get('scenario')}: joint={score.get('joint')} matches={score.get('matches')}/{score.get('gt_calls')} api_errors={len(row.get('api_errors') or [])} tools={names[:12]}")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def write_mini_report(run_id: str, mini_rows: List[Dict[str, Any]], mini_summary: Dict[str, Any], skipped: bool) -> Path:
    report = CODEX / "reports" / f"V37_V34_MINI_SHADOW_RESULT_{run_id}.md"
    by_scenario: Dict[str, List[Dict[str, Any]]] = {}
    for row in mini_rows:
        by_scenario.setdefault(str(row.get("scenario")), []).append(row.get("score", {}))
    lines = [
        "# V37 V34 Mini Shadow Result",
        "",
        f"- run_id: `{run_id}`",
        f"- skipped: {skipped}",
        f"- final run: False",
        f"- final hidden metadata used: False",
        "",
        "| set | valid | joint | result | tool | matched/gt | micro | calls |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        table_row("V37 mini selected", mini_summary),
        "",
        "## Per Scenario",
        "",
        "| scenario | valid | joint | result | tool | matched/gt | micro | calls |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for scenario, scores in sorted(by_scenario.items()):
        lines.append(table_row(scenario, aggregate(scores)))
    lines.extend(["", "## Per Task", ""])
    for row in mini_rows:
        score = row.get("score", {})
        names = [c.get("tool_name") for c in row.get("tool_program", [])]
        lines.append(f"- {row.get('task_key')} {row.get('variant')}: joint={score.get('joint')} result={score.get('result')} tool={score.get('tool')} matches={score.get('matches')}/{score.get('gt_calls')} calls={score.get('interaction_calls')} tools={names[:16]}")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- This is a compatibility mini shadow only. It is not a refreshed full V34 score.",
            "- Historical V34 11/41 should be treated as pre-sync reference until a full rerun under the synchronized official code.",
        ]
    )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default=f"v37_v34_chain_{stamp()}")
    ap.add_argument("--max-rounds", type=int, default=4)
    ap.add_argument("--max-tool-calls", type=int, default=24)
    ap.add_argument("--candidate-timeout-s", type=int, default=int(os.environ.get("V37_CANDIDATE_TIMEOUT_S", "180")))
    ap.add_argument("--skip-mini", action="store_true")
    args = ap.parse_args()

    analysis = CODEX / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    tasks = adapter.load_val41_tasks()
    v10_sha = sha256_file(V10_ZIP)

    smoke_a, ok_a = smoke_a_load(tasks, analysis / "v37_smoke_a_load_trace.jsonl")
    smoke_b, ok_b = smoke_b_dryrun(tasks, analysis / "v37_smoke_b_tool_dryrun.jsonl")
    smoke_c: List[Dict[str, Any]] = []
    ok_c = False
    if ok_a and ok_b:
        smoke_c, ok_c = smoke_c_native(tasks, analysis / "v37_smoke_c_v34_native_trace.jsonl", max_rounds=args.max_rounds, max_tool_calls=args.max_tool_calls)
    else:
        write_jsonl(analysis / "v37_smoke_c_v34_native_trace.jsonl", [])

    state = {
        "run_id": args.run_id,
        "version": "V37_OFFICIAL_SYNC_REBASE_AUDIT_AND_V34_CHAIN",
        "v10_zip_exists": V10_ZIP.exists(),
        "v10_sha256": v10_sha,
        "v10_expected_sha256": V10_EXPECTED_SHA,
        "v10_sha_matches_expected": v10_sha == V10_EXPECTED_SHA,
        "smoke_a_passed": ok_a,
        "smoke_b_passed": ok_b,
        "smoke_c_passed": ok_c,
        "final_run": False,
        "uses_final_hidden_metadata": False,
        "auto_submit": False,
    }
    smoke_report = write_smoke_report(args.run_id, state, smoke_a, smoke_b, smoke_c)
    state["smoke_report"] = str(smoke_report)

    mini_rows: List[Dict[str, Any]] = []
    mini_summary: Dict[str, Any] = aggregate([])
    skipped = True
    if ok_a and ok_b and ok_c and not args.skip_mini:
        skipped = False
        mini_rows, mini_summary = run_mini_shadow(
            tasks,
            analysis / "v37_v34_mini_shadow_results.jsonl",
            analysis / "v37_v34_mini_shadow_selection_trace.jsonl",
            max_rounds=args.max_rounds,
            max_tool_calls=args.max_tool_calls,
            timeout_s=args.candidate_timeout_s,
        )
    else:
        write_jsonl(analysis / "v37_v34_mini_shadow_results.jsonl", [])
        write_jsonl(analysis / "v37_v34_mini_shadow_selection_trace.jsonl", [])

    mini_report = write_mini_report(args.run_id, mini_rows, mini_summary, skipped=skipped)
    state.update(
        {
            "mini_shadow_skipped": skipped,
            "mini_shadow_summary": mini_summary,
            "mini_shadow_report": str(mini_report),
            "mini_shadow_task_count": len(mini_rows),
        }
    )
    write_json(CODEX / "state" / "latest_v37_v34_chain_smoke.json", state)
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
