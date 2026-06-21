#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Parallel V12 val41 runner for Track2.

Runs one subprocess per scenario spec. Each subprocess swaps only its own
official scenario JSON, so different specs can run concurrently.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
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
VERSION = "V12_official_style_qwen3vl_memory"
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
    path = SPLIT_DIR / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def spec_name(scenario: str, num: int) -> str:
    return f"{scenario}{num}"


def make_env(run_id: str, model: str, out_model: str) -> Dict[str, str]:
    env = os.environ.copy()
    env.update(load_shell_env_file(CODEX / "state" / ".openai_env"))
    base_url = (
        env.get("TRACK2_OPENAI_BASE_URL")
        or env.get("SERVICE_MODEL_API_BASE")
        or env.get("OPENAI_BASE_URL")
        or "https://ai-pixel.online/v1"
    )
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
    if not src.exists():
        raise FileNotFoundError(src)
    out_dir = CODEX / "runs" / "V8_tmp_scenarios" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / src.name
    shutil.copy2(src, dst)
    return dst


def run_spec(spec: Tuple[str, int, List[int]], run_id: str, model: str, timeout: int) -> Dict[str, Any]:
    scenario, num, idxs = spec
    name = spec_name(scenario, num)
    out_model = f"{model}-{VERSION}-{run_id}"
    env = make_env(run_id, model, out_model)
    log_dir = CODEX / "runs" / VERSION / run_id / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    result_path = EGO / "results" / out_model / f"{name}_easy.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = prepare_tmp(run_id, scenario, num)
    official = EGO / "scenarios" / "final" / f"{name}.json"
    backup = CODEX / "runs" / "V8_tmp_scenarios" / run_id / f"{name}.official_backup.json"
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
            log.write(f"[parallel] spec={name} tasks={len(idxs)} run_id={run_id}\n")
            log.flush()
            try:
                cp = subprocess.run(cmd, cwd=str(EGO), env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
                rc = cp.returncode
            except subprocess.TimeoutExpired:
                timed_out = True
                rc = 124
                log.write(f"\n[parallel] timed out after {timeout}s\n")
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
        scenario = item["scenario"]
        num = item["number"]
        gt_tmp = CODEX / "runs" / "V8_tmp_scenarios" / run_id / f"{scenario}{num}.json"
        result = Path(item["result_file"])
        if not result.exists():
            rows.append(
                {
                    "scenario": scenario,
                    "number": num,
                    "spec": item["spec"],
                    "valid": 0,
                    "joint": 0,
                    "result": 0,
                    "tool": 0,
                    "micro": 0,
                    "error": "missing_result",
                }
            )
            continue
        try:
            metrics = evaluate_interaction_success(
                str(gt_tmp),
                str(result),
                scenario=scenario,
                args=_argparse.Namespace(scenario_number=num),
                silent=True,
                num_samples=0,
            )
            micro_stats = metrics.get("micro_tool_stats", {}) or {}
            rows.append(
                {
                    "scenario": scenario,
                    "number": num,
                    "spec": item["spec"],
                    "valid": metrics.get("valid_scenarios", 0),
                    "joint": metrics.get("joint_success", {}).get("success_rate", 0),
                    "result": metrics.get("result_based", {}).get("success_rate", 0),
                    "tool": metrics.get("tool_based", {}).get("success_rate", 0),
                    "micro": micro_stats.get("micro_accuracy", 0),
                    "avg_task_accuracy": micro_stats.get("avg_task_accuracy", 0),
                    "correct_calls": micro_stats.get("total_correct_calls", 0),
                    "gt_calls": micro_stats.get("total_ground_truth_calls", 0),
                    "interaction_calls": micro_stats.get("total_interaction_calls", 0),
                    "error": "",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "scenario": scenario,
                    "number": num,
                    "spec": item["spec"],
                    "valid": 0,
                    "joint": 0,
                    "result": 0,
                    "tool": 0,
                    "micro": 0,
                    "error": f"{type(exc).__name__}: {str(exc)[:260]}",
                }
            )
    total_valid = sum(r.get("valid", 0) for r in rows)

    def wavg(key: str) -> float:
        return sum(r.get(key, 0) * r.get("valid", 0) for r in rows) / total_valid if total_valid else 0.0

    total_correct = sum(r.get("correct_calls", 0) for r in rows)
    total_gt = sum(r.get("gt_calls", 0) for r in rows)
    total_interaction = sum(r.get("interaction_calls", 0) for r in rows)
    return {
        "rows": rows,
        "summary": {
            "valid": total_valid,
            "joint": wavg("joint"),
            "result": wavg("result"),
            "tool": wavg("tool"),
            "micro": total_correct / total_gt if total_gt else wavg("micro"),
            "avg_task_accuracy": wavg("avg_task_accuracy"),
            "correct_calls": total_correct,
            "gt_calls": total_gt,
            "interaction_calls": total_interaction,
        },
    }


def audit_qwen_hits(run_id: str) -> Dict[str, Any]:
    root = CODEX / "runs" / VERSION / run_id / "qwen3vl_grounding_hits"
    events = []
    if root.exists():
        for path in sorted(root.glob("*.jsonl")):
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    item["_file"] = path.name
                    events.append(item)
                except Exception:
                    events.append({"_file": path.name, "parse_error": True})
    task_keys = set()
    topk_events = 0
    fallback_events = 0
    missing_events = 0
    status_counts: Dict[str, int] = {}
    for ev in events:
        key = f"{ev.get('scenario_spec') or ev.get('scenario') or ''}_{ev.get('task_id') or ''}"
        if key.strip("_"):
            task_keys.add(key)
        if int(ev.get("top_k_count") or 0) > 0:
            topk_events += 1
        if ev.get("video_fallback_used"):
            fallback_events += 1
        status = str(ev.get("cache_status") or ev.get("status") or ev.get("task_card_status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        if "missing" in status or "failed" in status:
            missing_events += 1
    return {
        "hit_dir": str(root),
        "event_count": len(events),
        "task_count": len(task_keys),
        "task_keys": sorted(task_keys),
        "events_with_top_k": topk_events,
        "video_fallback_events": fallback_events,
        "missing_or_failed_events": missing_events,
        "status_counts": status_counts,
    }


def write_report(run_id: str, model: str, run_items: List[Dict[str, Any]], eval_result: Dict[str, Any], qwen: Dict[str, Any], manifest: Dict[str, Any], max_workers: int) -> Path:
    report = CODEX / "reports" / f"V12_VAL41_PARALLEL_QWEN3VL_MEMORY_{run_id}.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    summary = eval_result["summary"]
    lines = [
        f"# V12 Val41 Parallel Qwen3-VL Memory {run_id}",
        "",
        f"- version: `{VERSION}`",
        f"- model: `{model}`",
        "- stage: `validation_A_medium`",
        f"- max_workers: {max_workers}",
        "- final_run: false",
        "- api_key_logged: false",
        "- v10_zip_overwritten: false",
        f"- materialized_split: `{SPLIT_DIR}`",
        f"- split_source: `{manifest.get('source')}`",
        "",
        "## Summary",
        "",
        f"- valid: {summary.get('valid', 0)}",
        f"- joint: {summary.get('joint', 0):.4f}",
        f"- result: {summary.get('result', 0):.4f}",
        f"- tool: {summary.get('tool', 0):.4f}",
        f"- micro: {summary.get('micro', 0):.4f}",
        f"- avg_task_accuracy: {summary.get('avg_task_accuracy', 0):.4f}",
        f"- tool_call_match_counts: {summary.get('correct_calls', 0)}/{summary.get('gt_calls', 0)} gt, interaction_calls={summary.get('interaction_calls', 0)}",
        "",
        "## Qwen3-VL Prompt Hit Audit",
        "",
        f"- hit_dir: `{qwen.get('hit_dir')}`",
        f"- qwen3vl_prompt_events: {qwen.get('event_count', 0)}",
        f"- qwen3vl_prompt_tasks: {qwen.get('task_count', 0)}/41",
        f"- qwen3vl_events_with_top_k: {qwen.get('events_with_top_k', 0)}",
        f"- qwen3vl_video_fallback_events: {qwen.get('video_fallback_events', 0)}",
        f"- qwen3vl_missing_or_failed_events: {qwen.get('missing_or_failed_events', 0)}",
        f"- qwen3vl_status_counts: `{json.dumps(qwen.get('status_counts', {}), ensure_ascii=False)}`",
        "",
        "## Per File Metrics",
        "",
        "| scenario | n | rc | valid | joint | result | tool | micro | calls | error |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    items_by_spec = {item["spec"]: item for item in run_items}
    for row in sorted(eval_result["rows"], key=lambda x: x["spec"]):
        item = items_by_spec.get(row["spec"], {})
        lines.append(
            f"| {row['spec']} | {len(item.get('indices', []))} | {item.get('returncode', '')} | "
            f"{row.get('valid', 0)} | {row.get('joint', 0):.3f} | {row.get('result', 0):.3f} | "
            f"{row.get('tool', 0):.3f} | {row.get('micro', 0):.3f} | "
            f"{row.get('correct_calls', 0)}/{row.get('gt_calls', 0)} | {row.get('error', '') or item.get('error', '')} |"
        )
    lines += ["", "## Run Items", ""]
    for item in sorted(run_items, key=lambda x: x["spec"]):
        lines.append(
            f"- {item['spec']} idx={item['indices']} rc={item['returncode']} seconds={item['seconds']} "
            f"result={item['result_file']} log={item['log']}"
        )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="")
    ap.add_argument("--model", default=os.environ.get("TRACK2_OPENAI_MODEL", "gpt-5.5"))
    ap.add_argument("--max-workers", type=int, default=int(os.environ.get("TRACK2_V12_VAL41_WORKERS", "10")))
    ap.add_argument("--timeout", type=int, default=int(os.environ.get("TRACK2_V12_SPEC_TIMEOUT", "1800")))
    args = ap.parse_args()
    run_id = args.run_id or f"V12_qwen3vl_prior_all_modules_val41_parallel_{time.strftime('%Y%m%d_%H%M%S')}"
    manifest = read_manifest()
    specs = [(s, int(n), [int(x) for x in idxs]) for s, n, idxs in manifest.get("specs", [])]
    out_dir = CODEX / "runs" / VERSION / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    run_manifest = {
        "stage": "validation_A_medium",
        "version": VERSION,
        "run_id": run_id,
        "model": args.model,
        "specs": specs,
        "max_workers": args.max_workers,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "split_manifest_sha256": hashlib.sha256((SPLIT_DIR / "manifest.json").read_bytes()).hexdigest(),
        "final_run": False,
    }
    (out_dir / "manifest.json").write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if not load_shell_env_file(CODEX / "state" / ".openai_env").get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY missing in state/.openai_env")
    run_items: List[Dict[str, Any]] = []
    with cf.ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as ex:
        futs = [ex.submit(run_spec, spec, run_id, args.model, args.timeout) for spec in specs]
        for fut in cf.as_completed(futs):
            item = fut.result()
            run_items.append(item)
            print(json.dumps({"finished": item["spec"], "rc": item["returncode"], "seconds": item["seconds"], "error": item["error"]}, ensure_ascii=False), flush=True)
            (out_dir / "run_items.partial.json").write_text(json.dumps(sorted(run_items, key=lambda x: x["spec"]), ensure_ascii=False, indent=2), encoding="utf-8")
    run_items = sorted(run_items, key=lambda x: x["spec"])
    (out_dir / "run_items.json").write_text(json.dumps(run_items, ensure_ascii=False, indent=2), encoding="utf-8")
    eval_result = evaluate_subset(run_items, run_id)
    (out_dir / "eval_summary.json").write_text(json.dumps(eval_result, ensure_ascii=False, indent=2), encoding="utf-8")
    qwen = audit_qwen_hits(run_id)
    (out_dir / "qwen3vl_prompt_hit_audit.json").write_text(json.dumps(qwen, ensure_ascii=False, indent=2), encoding="utf-8")
    report = write_report(run_id, args.model, run_items, eval_result, qwen, manifest, args.max_workers)
    latest = {
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "run_id": run_id,
        "version": VERSION,
        "model": args.model,
        "stage": "validation_A_medium",
        "parallel": True,
        "report": str(report),
        "summary": eval_result["summary"],
        "qwen3vl_prompt_hit_audit": qwen,
        "final_run": False,
        "v10_zip_overwritten": False,
    }
    (CODEX / "state" / "latest_v12_val41_parallel.json").write_text(json.dumps(latest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(latest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
