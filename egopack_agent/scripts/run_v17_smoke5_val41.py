#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V17 fail-fast smoke5 runner.

Runs only five frozen val41-derived materialized tasks after hard gates pass.
Val labels are used only by the evaluator after model outputs are written.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple


CODEX = Path(os.environ.get("CODEX_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/codex"))
EGO = Path(os.environ.get("EGO_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/EgoBench"))
PY = os.environ.get("TRACK2_PYTHON", "python3")
VERSION = "V17_GT100_EXECUTABLE_COMPILER_SMOKE5"
VAL_SPLIT = CODEX / "state" / "materialized_splits" / "validation_A_limit30"
RUNS = CODEX / "runs" / VERSION

SMOKE_SPECS = [
    ("order", 1, "largest order bottleneck"),
    ("restaurant", 3, "restaurant visual dish slot"),
    ("retail", 7, "retail visual product plus condition"),
    ("retail", 8, "retail relative-location cart mutation"),
    ("retail", 2, "fallback near-miss slot"),
]


def load_shell_env_file(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or not line.startswith("export ") or "=" not in line:
            continue
        key, value = line[len("export "):].split("=", 1)
        out[key.strip()] = value.strip().strip('"').strip("'")
    return out


def ensure_hard_gates() -> Dict[str, Any]:
    gt_manifest = json.loads((CODEX / "gt_distill_v17" / "gt100_verified_manifest.json").read_text(encoding="utf-8"))
    gpt_manifest = json.loads((CODEX / "gt_distill_v17" / "gpt55_distillation_manifest.json").read_text(encoding="utf-8"))
    rules_path = CODEX / "gt_distill_v17" / "gpt55_distilled_rules.jsonl"
    valid_rows = 0
    scenario_counts: Dict[str, int] = {}
    with rules_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("valid") is True:
                valid_rows += 1
                scenario_counts[row.get("scenario") or "unknown"] = scenario_counts.get(row.get("scenario") or "unknown", 0) + 1
    errors = []
    if gt_manifest.get("status") != "PASS":
        errors.append("gt100_manifest_not_pass")
    if gpt_manifest.get("status") != "PASS":
        errors.append("gpt55_manifest_not_pass")
    if valid_rows < 600:
        errors.append(f"valid_gpt_rows_lt_600:{valid_rows}")
    if float(gpt_manifest.get("parse_valid_rate") or 0) < 0.95:
        errors.append(f"parse_valid_rate_lt_095:{gpt_manifest.get('parse_valid_rate')}")
    missing = sorted(set(["order", "restaurant", "retail", "kitchen"]) - set(scenario_counts))
    if missing:
        errors.append("missing_scenarios:" + ",".join(missing))
    if errors:
        path = CODEX / "reports" / f"NEED_HUMAN_ATTENTION_V17_HARD_GATE_{time.strftime('%Y%m%d_%H%M%S')}.md"
        path.write_text("# V17 Hard Gate Failed\n\n" + "\n".join(f"- {e}" for e in errors) + "\n", encoding="utf-8")
        raise SystemExit(f"V17 hard gate failed, see {path}")
    return {
        "gt100_status": gt_manifest.get("status"),
        "gpt55_status": gpt_manifest.get("status"),
        "valid_gpt_rows": valid_rows,
        "parse_valid_rate": gpt_manifest.get("parse_valid_rate"),
        "scenario_counts": scenario_counts,
    }


def materialize_smoke5(run_id: str) -> Tuple[Path, List[Dict[str, Any]]]:
    manifest = json.loads((VAL_SPLIT / "manifest.json").read_text(encoding="utf-8"))
    by_spec = {(str(s), int(n)): [int(x) for x in idxs] for s, n, idxs in manifest.get("specs", [])}
    out_dir = CODEX / "state" / "materialized_splits" / f"v17_smoke5_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    items: List[Dict[str, Any]] = []
    for scenario, number, reason in SMOKE_SPECS:
        spec = f"{scenario}{number}"
        src = VAL_SPLIT / f"{spec}.json"
        if not src.exists():
            raise SystemExit(f"missing source split {src}")
        rows = json.loads(src.read_text(encoding="utf-8"))
        if not rows:
            raise SystemExit(f"empty source split {src}")
        dst = out_dir / f"{spec}.json"
        dst.write_text(json.dumps([rows[0]], ensure_ascii=False, indent=2), encoding="utf-8")
        orig_indices = by_spec.get((scenario, number), [])
        items.append({
            "scenario": scenario,
            "number": number,
            "spec": spec,
            "task_count": 1,
            "source_original_index": orig_indices[0] if orig_indices else None,
            "source_position_in_materialized_file": 0,
            "reason": reason,
            "file": str(dst),
        })
    manifest_out = {
        "run_id": run_id,
        "version": VERSION,
        "source_split": str(VAL_SPLIT),
        "uses_val41_gt_for_policy": False,
        "uses_val41_gt_for_post_eval_only": True,
        "uses_final_hidden_metadata": False,
        "items": items,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest_out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_dir, items


def base_env(run_id: str, model: str, out_model: str, candidate: str) -> Dict[str, str]:
    env = os.environ.copy()
    env.update(load_shell_env_file(CODEX / "state" / ".openai_env"))
    base_url = env.get("TRACK2_OPENAI_BASE_URL") or env.get("SERVICE_MODEL_API_BASE") or "https://ai-pixel.online/v1"
    api_key = env.get("OPENAI_API_KEY", "")
    no_proxy = "ai-pixel.online,cf.ai-pixel.online,localhost,127.0.0.1,::1"
    env.update({
        "CODEX_ROOT": str(CODEX),
        "EGO_ROOT": str(EGO),
        "TRACK2_OPENAI_BASE_URL": base_url,
        "SERVICE_MODEL_API_BASE": base_url,
        "SERVICE_MODEL_API_KEY": api_key,
        "USER_AGENT_API_BASE_URL": base_url,
        "USER_AGENT_API_KEY": api_key,
        "SERVICE_MODEL_BACKEND": "openai_compatible_chat",
        "SERVICE_MODEL_NAME": model,
        "USER_MODEL_NAME": model,
        "TRACK2_USER_USE_OPENAI": "0",
        "TRACK2_USE_OPENAI_GPT55": "0",
        "TRACK2_GPT55_STRUCTURED_OUTPUT": "0",
        "TRACK2_DEFAULT_MAX_TOKENS": env.get("TRACK2_DEFAULT_MAX_TOKENS", "2048"),
        "TRACK2_CONNECT_TIMEOUT": env.get("TRACK2_CONNECT_TIMEOUT", "10"),
        "TRACK2_READ_TIMEOUT": env.get("TRACK2_READ_TIMEOUT", "240"),
        "TRACK2_API_MAX_RETRIES": env.get("TRACK2_API_MAX_RETRIES", "1"),
        "TRACK2_TEMPERATURE": env.get("TRACK2_TEMPERATURE", "0.05"),
        "TRACK2_MAX_TURNS": env.get("TRACK2_MAX_TURNS", "6"),
        "TRACK2_V8_TASK_TIMEOUT": env.get("TRACK2_V8_TASK_TIMEOUT", "1800"),
        "TRACK2_USE_VIDEO": "0",
        "TRACK2_TEXT_ONLY_VISUAL_CONTEXT": "1",
        "TRACK2_ENABLE_VISUAL_CACHE": "1",
        "TRACK2_ENABLE_OFFICIAL_STYLE_PROMPT": "1",
        "TRACK2_ENABLE_DB_GUARD": "1",
        "TRACK2_ENABLE_PLANNER": "1",
        "TRACK2_ENABLE_SCENARIO_RULES": "1",
        "TRACK2_ENABLE_EVALUATOR_AWARENESS": "1",
        "TRACK2_ENABLE_MEMORY_RETRIEVAL": "1",
        "TRACK2_MEMORY_BANK_DIR": str(CODEX / "memory_bank_v10"),
        "TRACK2_ENABLE_VISUAL_GROUNDING_RESOLVER": "1",
        "TRACK2_ENABLE_QWEN3VL_GROUNDING": "1",
        "TRACK2_QWEN3VL_GROUNDING_DIR": str(CODEX / "visual_cache_v12" / "qwen3vl_grounding"),
        "TRACK2_QWEN3VL_VIDEO_GROUNDING_DIR": str(CODEX / "visual_cache_v12" / "qwen3vl_grounding_by_video"),
        "TRACK2_ENABLE_RETAIL_NARROWER": "1",
        "TRACK2_ENABLE_RETAIL_CANDIDATE_NARROWER": "1",
        "TRACK2_ENABLE_RETAIL_PROCESS_TRIMMER": "1",
        "TRACK2_ENABLE_ORDER_PROCESS_MEMORY": "1",
        "TRACK2_ENABLE_ORDER_PROCESS_SYNTHESIS": "1",
        "TRACK2_ENABLE_V9_SOFT_GUARD": "1",
        "TRACK2_ENABLE_SOFT_GUARD": "1",
        "TRACK2_ENABLE_MULTICANDIDATE": "1",
        "TRACK2_ENABLE_MULTICANDIDATE_RERANK": "1",
        "TRACK2_ENABLE_DEEPSEEK_CROSSCHECK": "0",
        "TRACK2_USE_DEEPSEEK_CROSSCHECK": "0",
        "TRACK2_ENABLE_V14_PROCESS_POLICY": "0",
        "TRACK2_ENABLE_V16_PROCESS_POLICY": "0",
        "TRACK2_ENABLE_V17_COMPILER": "0",
        "TRACK2_V17_EXEC_REPAIR": "0",
        "TRACK2_V17_DISTILL_DIR": str(CODEX / "gt_distill_v17"),
        "TRACK2_RUN_VERSION": candidate,
        "TRACK2_RUN_ID": run_id,
        "TRACK2_OUTPUT_MODEL_NAME": out_model,
        "TRACK2_FINAL_EVAL": "0",
        "NO_PROXY": no_proxy,
        "no_proxy": no_proxy,
        "PYTHONPATH": f"{CODEX}/wrappers:{CODEX}:" + env.get("PYTHONPATH", ""),
    })
    if candidate.startswith("V16_"):
        env["TRACK2_ENABLE_V16_PROCESS_POLICY"] = "1"
        env["TRACK2_V16_DISTILL_DIR"] = str(CODEX / "gt_distill_v16")
    if candidate.startswith("V17_"):
        env["TRACK2_ENABLE_V17_COMPILER"] = "1"
        env["TRACK2_ENABLE_V16_PROCESS_POLICY"] = "0"
    if "repaired" in candidate:
        env["TRACK2_V17_EXEC_REPAIR"] = "1"
    for key in ["HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "https_proxy", "http_proxy", "all_proxy"]:
        env.pop(key, None)
    return env


def run_one_spec(item: Dict[str, Any], smoke_dir: Path, run_id: str, model: str, candidate: str, timeout: int) -> Dict[str, Any]:
    spec = item["spec"]
    scenario = item["scenario"]
    number = int(item["number"])
    out_model = f"{model}-{candidate}-{run_id}"
    env = base_env(run_id, model, out_model, candidate)
    log_dir = RUNS / run_id / candidate / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    tmp = smoke_dir / f"{spec}.json"
    official = EGO / "scenarios" / "final" / f"{spec}.json"
    backup = RUNS / run_id / "official_backups" / f"{spec}.backup.json"
    backup.parent.mkdir(parents=True, exist_ok=True)
    result_path = EGO / "results" / out_model / f"{spec}_easy.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        PY,
        str(CODEX / "runners" / "track2_multi_agent_plus.py"),
        "--scenario",
        scenario,
        "--scenario_number",
        str(number),
        "--service_model_name",
        model,
        "--num_tasks",
        "1",
    ]
    started = time.time()
    rc = 999
    timed_out = False
    error = ""
    log_path = log_dir / f"{spec}.log"
    try:
        shutil.copy2(official, backup)
        shutil.copy2(tmp, official)
        with log_path.open("w", encoding="utf-8") as log:
            log.write(f"[v17 smoke5] candidate={candidate} spec={spec} run_id={run_id}\n")
            log.flush()
            try:
                cp = subprocess.run(cmd, cwd=str(EGO), env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
                rc = cp.returncode
            except subprocess.TimeoutExpired:
                timed_out = True
                rc = 124
                log.write(f"\n[v17 smoke5] timed out after {timeout}s\n")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            if backup.exists():
                shutil.copy2(backup, official)
        except Exception as exc:
            error = (error + "; " if error else "") + f"restore_error={type(exc).__name__}: {exc}"
    return {
        **item,
        "candidate": candidate,
        "returncode": rc,
        "timed_out": timed_out,
        "error": error,
        "seconds": round(time.time() - started, 2),
        "output_model": out_model,
        "result_file": str(result_path),
        "log": str(log_path),
    }


def evaluate_items(run_items: List[Dict[str, Any]], smoke_dir: Path) -> Dict[str, Any]:
    sys.path.insert(0, str(EGO / "analysis_scripts"))
    sys.path.insert(0, str(EGO))
    from evaluate_interaction import evaluate_interaction_success
    import argparse as _argparse

    rows = []
    for item in sorted(run_items, key=lambda x: (x["candidate"], x["spec"])):
        result = Path(item["result_file"])
        gt_tmp = smoke_dir / f"{item['spec']}.json"
        if not result.exists():
            rows.append({**item, "valid": 0, "joint": 0, "result": 0, "tool": 0, "micro": 0, "error": item.get("error") or "missing_result"})
            continue
        try:
            metrics = evaluate_interaction_success(
                str(gt_tmp),
                str(result),
                scenario=item["scenario"],
                args=_argparse.Namespace(scenario_number=int(item["number"])),
                silent=True,
                num_samples=0,
            )
            micro = metrics.get("micro_tool_stats", {}) or {}
            rows.append({
                **item,
                "valid": metrics.get("valid_scenarios", 0),
                "joint": metrics.get("joint_success", {}).get("success_rate", 0),
                "result": metrics.get("result_based", {}).get("success_rate", 0),
                "tool": metrics.get("tool_based", {}).get("success_rate", 0),
                "micro": micro.get("micro_accuracy", 0),
                "avg_task_accuracy": micro.get("avg_task_accuracy", 0),
                "correct_calls": micro.get("total_correct_calls", 0),
                "gt_calls": micro.get("total_ground_truth_calls", 0),
                "interaction_calls": micro.get("total_interaction_calls", 0),
                "eval_error": "",
            })
        except Exception as exc:
            rows.append({**item, "valid": 0, "joint": 0, "result": 0, "tool": 0, "micro": 0, "eval_error": f"{type(exc).__name__}: {exc}"})
    by_candidate: Dict[str, Dict[str, Any]] = {}
    for cand in sorted({r["candidate"] for r in rows}):
        cr = [r for r in rows if r["candidate"] == cand]
        valid = sum(r.get("valid", 0) for r in cr)
        correct = sum(r.get("correct_calls", 0) for r in cr)
        gt = sum(r.get("gt_calls", 0) for r in cr)
        def wavg(k: str) -> float:
            return sum(r.get(k, 0) * r.get("valid", 0) for r in cr) / valid if valid else 0.0
        by_candidate[cand] = {
            "valid": valid,
            "joint": wavg("joint"),
            "result": wavg("result"),
            "tool": wavg("tool"),
            "micro": correct / gt if gt else wavg("micro"),
            "avg_task_accuracy": wavg("avg_task_accuracy"),
            "correct_calls": correct,
            "gt_calls": gt,
            "interaction_calls": sum(r.get("interaction_calls", 0) for r in cr),
            "task_joint_count": sum(1 for r in cr if r.get("joint", 0) >= 1.0),
        }
    return {"rows": rows, "by_candidate": by_candidate}


def write_reports(run_id: str, hard_gate: Dict[str, Any], smoke_dir: Path, items: List[Dict[str, Any]], eval_result: Dict[str, Any]) -> Tuple[Path, Path]:
    report = CODEX / "reports" / f"V17_SMOKE5_VAL41_{run_id}.md"
    by = eval_result["by_candidate"]
    v16 = by.get("V16_gt100_distilled_policy_smoke5", {})
    v17 = by.get("V17_compiler_repaired_smoke5", {})
    v16_micro = float(v16.get("micro") or 0)
    v17_micro = float(v17.get("micro") or 0)
    v16_joint = int(v16.get("task_joint_count") or 0)
    v17_joint = int(v17.get("task_joint_count") or 0)
    order_row = None
    for row in eval_result["rows"]:
        if row["candidate"] == "V17_compiler_repaired_smoke5" and row["spec"] == "order1":
            order_row = row
            break
    order_correct = int((order_row or {}).get("correct_calls") or 0)
    order_gt = int((order_row or {}).get("gt_calls") or 0)
    passed = (v17_joint > v16_joint) or ((v17_micro - v16_micro) >= 0.10) or (order_gt >= 24 and order_correct >= 8)
    lines = [
        f"# V17 Smoke5 Val41 {run_id}",
        "",
        "- version: V17_GT100_EXECUTABLE_COMPILER_SMOKE5",
        "- final_run: false",
        "- v10_zip_overwritten: false",
        "- auto_submit: false",
        "- uses_final_hidden_metadata: false",
        "- uses_val41_gt_for_policy: false",
        "- uses_val41_gt_for_post_eval_only: true",
        f"- smoke_success_gate_passed: {str(passed).lower()}",
        "",
        "## Hard Gates",
        "",
        f"- GT100 manifest: {hard_gate.get('gt100_status')}",
        f"- GPT-5.5 distillation manifest: {hard_gate.get('gpt55_status')}",
        f"- valid_gpt_rows: {hard_gate.get('valid_gpt_rows')}",
        f"- parse_valid_rate: {hard_gate.get('parse_valid_rate')}",
        f"- scenarios: {json.dumps(hard_gate.get('scenario_counts'), ensure_ascii=False)}",
        "",
        "## Smoke Items",
        "",
        "| spec | source_original_index | reason |",
        "|---|---:|---|",
    ]
    for item in items:
        lines.append(f"| {item['spec']} | {item.get('source_original_index')} | {item['reason']} |")
    lines += [
        "",
        "## Candidate Summary",
        "",
        "| candidate | valid | joint | result | tool | micro | calls | joint_tasks |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cand, s in sorted(by.items()):
        lines.append(f"| {cand} | {s.get('valid', 0)} | {s.get('joint', 0):.3f} | {s.get('result', 0):.3f} | {s.get('tool', 0):.3f} | {s.get('micro', 0):.3f} | {s.get('correct_calls', 0)}/{s.get('gt_calls', 0)} | {s.get('task_joint_count', 0)} |")
    lines += [
        "",
        "## Per Task",
        "",
        "| candidate | spec | rc | joint | result | tool | micro | calls | interaction_calls | error |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in sorted(eval_result["rows"], key=lambda r: (r["candidate"], r["spec"])):
        err = row.get("eval_error") or row.get("error") or ""
        lines.append(f"| {row['candidate']} | {row['spec']} | {row.get('returncode', '')} | {row.get('joint', 0):.3f} | {row.get('result', 0):.3f} | {row.get('tool', 0):.3f} | {row.get('micro', 0):.3f} | {row.get('correct_calls', 0)}/{row.get('gt_calls', 0)} | {row.get('interaction_calls', 0)} | {err} |")
    lines += [
        "",
        "## Decision Inputs",
        "",
        f"- V16 smoke joint_tasks: {v16_joint}",
        f"- V17 repaired smoke joint_tasks: {v17_joint}",
        f"- V16 smoke micro: {v16_micro:.4f}",
        f"- V17 repaired smoke micro: {v17_micro:.4f}",
        f"- order1 V17 repaired calls: {order_correct}/{order_gt}",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    decision = CODEX / "reports" / f"V17_NEXT_DECISION_{run_id}.md"
    decision_lines = [
        f"# V17 Next Decision {run_id}",
        "",
        f"- smoke_success_gate_passed: {str(passed).lower()}",
        "- run_final: false",
        "- run_full_val41_now: false",
        "- protected_best_updated: false",
        f"- smoke_report: `{report}`",
        "",
    ]
    if passed:
        decision_lines.append("Decision: V17 smoke met the minimum continuation gate. Next step can be full frozen val41 only if the user requests it.")
    else:
        decision_lines.append("Decision: V17 smoke did not meet the continuation gate. Stop here; do not expand to full val41 or final.")
    decision.write_text("\n".join(decision_lines) + "\n", encoding="utf-8")
    return report, decision


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default=f"v17_smoke5_{time.strftime('%Y%m%d_%H%M%S')}")
    ap.add_argument("--model", default=os.environ.get("TRACK2_OPENAI_MODEL", "gpt-5.5"))
    ap.add_argument("--max-workers", type=int, default=5)
    ap.add_argument("--timeout", type=int, default=1200)
    args = ap.parse_args()
    if not load_shell_env_file(CODEX / "state" / ".openai_env").get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY missing in state/.openai_env")
    hard_gate = ensure_hard_gates()
    smoke_dir, items = materialize_smoke5(args.run_id)
    candidates = [
        "V10_full_memory_smoke5",
        "V16_gt100_distilled_policy_smoke5",
        "V17_compiler_prompt_only_smoke5",
        "V17_compiler_repaired_smoke5",
    ]
    out_dir = RUNS / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps({
        "run_id": args.run_id,
        "version": VERSION,
        "model": args.model,
        "candidates": candidates,
        "smoke_dir": str(smoke_dir),
        "items": items,
        "hard_gate": hard_gate,
        "final_run": False,
        "uses_val41_gt_for_policy": False,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    run_items: List[Dict[str, Any]] = []
    for candidate in candidates:
        print(json.dumps({"candidate_start": candidate, "run_id": args.run_id}, ensure_ascii=False), flush=True)
        with cf.ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as ex:
            futs = [ex.submit(run_one_spec, item, smoke_dir, args.run_id, args.model, candidate, args.timeout) for item in items]
            for fut in cf.as_completed(futs):
                item = fut.result()
                run_items.append(item)
                print(json.dumps({"finished": item["candidate"], "spec": item["spec"], "rc": item["returncode"], "seconds": item["seconds"], "error": item["error"]}, ensure_ascii=False), flush=True)
                (out_dir / "run_items.partial.json").write_text(json.dumps(run_items, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "run_items.json").write_text(json.dumps(run_items, ensure_ascii=False, indent=2), encoding="utf-8")
    eval_result = evaluate_items(run_items, smoke_dir)
    (out_dir / "eval_summary.json").write_text(json.dumps(eval_result, ensure_ascii=False, indent=2), encoding="utf-8")
    report, decision = write_reports(args.run_id, hard_gate, smoke_dir, items, eval_result)
    state = {
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "run_id": args.run_id,
        "version": VERSION,
        "report": str(report),
        "decision_report": str(decision),
        "summary": eval_result["by_candidate"],
        "final_run": False,
        "protected_best_updated": False,
    }
    (CODEX / "state" / "latest_v17_smoke5.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(state, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
