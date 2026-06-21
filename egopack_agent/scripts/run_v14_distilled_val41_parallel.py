#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Parallel V14 distilled/no-task-oracle val41 runner."""
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
VERSION = "V14_val41_distilled_no_task_oracle"
SPLIT_DIR = CODEX / "state" / "materialized_splits" / "validation_A_limit30"


def load_shell_env_file(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or not line.startswith("export "):
            continue
        body = line[len("export ") :]
        if "=" not in body:
            continue
        key, value = body.split("=", 1)
        out[key.strip()] = value.strip().strip('"').strip("'")
    return out


def read_manifest() -> Dict[str, Any]:
    return json.loads((SPLIT_DIR / "manifest.json").read_text(encoding="utf-8"))


def make_env(run_id: str, model: str, out_model: str) -> Dict[str, str]:
    env = os.environ.copy()
    env.update(load_shell_env_file(CODEX / "state" / ".openai_env"))
    base_url = env.get("TRACK2_OPENAI_BASE_URL") or env.get("SERVICE_MODEL_API_BASE") or "https://ai-pixel.online/v1"
    api_key = env.get("OPENAI_API_KEY", "")
    no_proxy = "ai-pixel.online,cf.ai-pixel.online,localhost,127.0.0.1,::1"
    env.update(
        {
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
            "TRACK2_TEMPERATURE": env.get("TRACK2_TEMPERATURE", "0.1"),
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
            "TRACK2_ENABLE_V14_PROCESS_POLICY": "1",
            "TRACK2_V14_MEMORY_BANK_DIR": str(CODEX / "memory_bank_v14_gt_trajectory"),
            "TRACK2_RUN_VERSION": VERSION,
            "TRACK2_RUN_ID": run_id,
            "TRACK2_OUTPUT_MODEL_NAME": out_model,
            "NO_PROXY": no_proxy,
            "no_proxy": no_proxy,
            "PYTHONPATH": f"{CODEX}/wrappers:{CODEX}:" + env.get("PYTHONPATH", ""),
        }
    )
    for k in ["HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "https_proxy", "http_proxy", "all_proxy"]:
        env.pop(k, None)
    return env


def prepare_tmp(run_id: str, scenario: str, num: int) -> Path:
    src = SPLIT_DIR / f"{scenario}{num}.json"
    out_dir = CODEX / "runs" / "V8_tmp_scenarios" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / src.name
    shutil.copy2(src, dst)
    return dst


def run_spec(spec: Tuple[str, int, List[int]], run_id: str, model: str, timeout: int) -> Dict[str, Any]:
    scenario, num, idxs = spec
    name = f"{scenario}{num}"
    out_model = f"{model}-{VERSION}-{run_id}"
    env = make_env(run_id, model, out_model)
    log_dir = CODEX / "runs" / VERSION / run_id / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    tmp = prepare_tmp(run_id, scenario, num)
    official = EGO / "scenarios" / "final" / f"{name}.json"
    backup = CODEX / "runs" / "V8_tmp_scenarios" / run_id / f"{name}.official_backup.json"
    result_path = EGO / "results" / out_model / f"{name}_easy.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        PY,
        str(CODEX / "runners" / "track2_multi_agent_plus.py"),
        "--scenario",
        scenario,
        "--scenario_number",
        str(num),
        "--service_model_name",
        model,
        "--num_tasks",
        str(len(json.loads(tmp.read_text(encoding="utf-8")))),
    ]
    started = time.time()
    rc = 999
    timed_out = False
    error = ""
    log_path = log_dir / f"{name}.log"
    try:
        shutil.copy2(official, backup)
        shutil.copy2(tmp, official)
        with log_path.open("w", encoding="utf-8") as log:
            log.write(f"[v14 parallel] spec={name} tasks={len(idxs)} run_id={run_id}\n")
            log.flush()
            try:
                cp = subprocess.run(cmd, cwd=str(EGO), env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
                rc = cp.returncode
            except subprocess.TimeoutExpired:
                timed_out = True
                rc = 124
                log.write(f"\n[v14 parallel] timed out after {timeout}s\n")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            if backup.exists():
                shutil.copy2(backup, official)
        except Exception as exc:
            error = (error + "; " if error else "") + f"restore_error={type(exc).__name__}: {exc}"
    return {
        "scenario": scenario,
        "number": num,
        "spec": name,
        "indices": idxs,
        "returncode": rc,
        "timed_out": timed_out,
        "error": error,
        "seconds": round(time.time() - started, 2),
        "output_model": out_model,
        "result_file": str(result_path),
        "log": str(log_path),
    }


def evaluate_subset(run_items: List[Dict[str, Any]], run_id: str) -> Dict[str, Any]:
    sys.path.insert(0, str(EGO / "analysis_scripts"))
    sys.path.insert(0, str(EGO))
    from evaluate_interaction import evaluate_interaction_success
    import argparse as _argparse

    rows = []
    for item in sorted(run_items, key=lambda x: x["spec"]):
        gt_tmp = CODEX / "runs" / "V8_tmp_scenarios" / run_id / f"{item['spec']}.json"
        result = Path(item["result_file"])
        if not result.exists():
            rows.append({"spec": item["spec"], "valid": 0, "joint": 0, "result": 0, "tool": 0, "micro": 0, "error": "missing_result"})
            continue
        try:
            metrics = evaluate_interaction_success(
                str(gt_tmp),
                str(result),
                scenario=item["scenario"],
                args=_argparse.Namespace(scenario_number=item["number"]),
                silent=True,
                num_samples=0,
            )
            micro = metrics.get("micro_tool_stats", {}) or {}
            rows.append({
                "spec": item["spec"],
                "scenario": item["scenario"],
                "number": item["number"],
                "valid": metrics.get("valid_scenarios", 0),
                "joint": metrics.get("joint_success", {}).get("success_rate", 0),
                "result": metrics.get("result_based", {}).get("success_rate", 0),
                "tool": metrics.get("tool_based", {}).get("success_rate", 0),
                "micro": micro.get("micro_accuracy", 0),
                "avg_task_accuracy": micro.get("avg_task_accuracy", 0),
                "correct_calls": micro.get("total_correct_calls", 0),
                "gt_calls": micro.get("total_ground_truth_calls", 0),
                "interaction_calls": micro.get("total_interaction_calls", 0),
                "error": "",
            })
        except Exception as exc:
            rows.append({"spec": item["spec"], "valid": 0, "joint": 0, "result": 0, "tool": 0, "micro": 0, "error": f"{type(exc).__name__}: {exc}"})
    total_valid = sum(r.get("valid", 0) for r in rows)
    def wavg(key: str) -> float:
        return sum(r.get(key, 0) * r.get("valid", 0) for r in rows) / total_valid if total_valid else 0
    correct = sum(r.get("correct_calls", 0) for r in rows)
    gt_calls = sum(r.get("gt_calls", 0) for r in rows)
    interaction = sum(r.get("interaction_calls", 0) for r in rows)
    return {
        "rows": rows,
        "summary": {
            "valid": total_valid,
            "joint": wavg("joint"),
            "result": wavg("result"),
            "tool": wavg("tool"),
            "micro": correct / gt_calls if gt_calls else wavg("micro"),
            "avg_task_accuracy": wavg("avg_task_accuracy"),
            "correct_calls": correct,
            "gt_calls": gt_calls,
            "interaction_calls": interaction,
        },
    }


def write_report(run_id: str, model: str, run_items: List[Dict[str, Any]], eval_result: Dict[str, Any]) -> Path:
    report = CODEX / "reports" / f"V14_DISTILLED_PROCESS_VAL41_{run_id}.md"
    s = eval_result["summary"]
    lines = [
        f"# V14 Distilled Process Val41 {run_id}",
        "",
        "- tier: B",
        "- uses_val41_gt_directly: false",
        "- uses_distilled_process_bank: true",
        "- final_run: false",
        "- v10_zip_overwritten: false",
        "",
        "## Summary",
        "",
        f"- valid: {s.get('valid', 0)}",
        f"- joint: {s.get('joint', 0):.4f}",
        f"- result: {s.get('result', 0):.4f}",
        f"- tool: {s.get('tool', 0):.4f}",
        f"- micro: {s.get('micro', 0):.4f}",
        f"- avg_task_accuracy: {s.get('avg_task_accuracy', 0):.4f}",
        f"- tool_call_match_counts: {s.get('correct_calls', 0)}/{s.get('gt_calls', 0)} gt, interaction_calls={s.get('interaction_calls', 0)}",
        "",
        "## Per File",
        "",
        "| spec | n | rc | valid | joint | result | tool | micro | calls | error |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    by_spec = {x["spec"]: x for x in run_items}
    for row in sorted(eval_result["rows"], key=lambda x: x["spec"]):
        item = by_spec.get(row["spec"], {})
        lines.append(f"| {row['spec']} | {len(item.get('indices', []))} | {item.get('returncode', '')} | {row.get('valid', 0)} | {row.get('joint', 0):.3f} | {row.get('result', 0):.3f} | {row.get('tool', 0):.3f} | {row.get('micro', 0):.3f} | {row.get('correct_calls', 0)}/{row.get('gt_calls', 0)} | {row.get('error', '') or item.get('error', '')} |")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default=f"v14_distilled_val41_{time.strftime('%Y%m%d_%H%M%S')}")
    ap.add_argument("--model", default=os.environ.get("TRACK2_OPENAI_MODEL", "gpt-5.5"))
    ap.add_argument("--max-workers", type=int, default=int(os.environ.get("TRACK2_V14_WORKERS", "10")))
    ap.add_argument("--timeout", type=int, default=int(os.environ.get("TRACK2_V14_SPEC_TIMEOUT", "1800")))
    args = ap.parse_args()
    manifest = read_manifest()
    specs = [(s, int(n), [int(x) for x in idxs]) for s, n, idxs in manifest.get("specs", [])]
    out_dir = CODEX / "runs" / VERSION / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps({"run_id": args.run_id, "version": VERSION, "specs": specs, "model": args.model, "workers": args.max_workers, "final_run": False}, ensure_ascii=False, indent=2), encoding="utf-8")
    if not load_shell_env_file(CODEX / "state" / ".openai_env").get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY missing in state/.openai_env")
    run_items: List[Dict[str, Any]] = []
    with cf.ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as ex:
        futs = [ex.submit(run_spec, spec, args.run_id, args.model, args.timeout) for spec in specs]
        for fut in cf.as_completed(futs):
            item = fut.result()
            run_items.append(item)
            print(json.dumps({"finished": item["spec"], "rc": item["returncode"], "seconds": item["seconds"], "error": item["error"]}, ensure_ascii=False), flush=True)
            (out_dir / "run_items.partial.json").write_text(json.dumps(sorted(run_items, key=lambda x: x["spec"]), ensure_ascii=False, indent=2), encoding="utf-8")
    run_items = sorted(run_items, key=lambda x: x["spec"])
    (out_dir / "run_items.json").write_text(json.dumps(run_items, ensure_ascii=False, indent=2), encoding="utf-8")
    eval_result = evaluate_subset(run_items, args.run_id)
    (out_dir / "eval_summary.json").write_text(json.dumps(eval_result, ensure_ascii=False, indent=2), encoding="utf-8")
    report = write_report(args.run_id, args.model, run_items, eval_result)
    state = {"updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "run_id": args.run_id, "version": VERSION, "model": args.model, "report": str(report), "summary": eval_result["summary"], "final_run": False, "uses_val41_gt_directly": False}
    (CODEX / "state" / "latest_v14_distilled_val41.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(state, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
