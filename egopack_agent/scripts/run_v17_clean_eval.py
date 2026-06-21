#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run V17 repaired compiler on the audited clean val41 subset only.

This script does not run final and does not use final hidden metadata for
policy. It temporarily swaps the official scenario file path because the
EgoBench runner expects that location, then restores the original file.
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
VERSION = "V17_clean_repaired_eval"
RUN_VERSION = "V17_compiler_repaired_clean"


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return default


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


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


def latest_clean_split() -> Path:
    state = read_json(CODEX / "state" / "latest_val41_clean_audit.json", {})
    clean = state.get("clean_split")
    if clean:
        path = Path(clean)
        if path.exists():
            return path
    candidates = sorted((CODEX / "state" / "materialized_splits").glob("validation_A_clean_*"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise SystemExit("No validation_A_clean_* split found. Run val41_clean_audit_and_eval.py first.")
    return candidates[-1]


def load_specs(clean_dir: Path) -> List[Tuple[str, int, str, int]]:
    manifest = read_json(clean_dir / "manifest.json", {})
    specs: List[Tuple[str, int, str, int]] = []
    for info in manifest.get("files", []):
        spec = Path(info["file"]).stem
        scenario = str(info.get("scenario") or "".join(ch for ch in spec if not ch.isdigit()))
        number = int(info.get("number") or spec[len(scenario):])
        task_count = int(info.get("task_count") or len(read_json(clean_dir / info["file"], [])))
        if task_count > 0:
            specs.append((scenario, number, spec, task_count))
    if not specs:
        raise SystemExit(f"No clean spec files found in {clean_dir}")
    return specs


def make_env(run_id: str, model: str, out_model: str) -> Dict[str, str]:
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
        "TRACK2_ENABLE_V17_COMPILER": "1",
        "TRACK2_V17_EXEC_REPAIR": "1",
        "TRACK2_V17_DISTILL_DIR": str(CODEX / "gt_distill_v17"),
        "TRACK2_RUN_VERSION": RUN_VERSION,
        "TRACK2_RUN_ID": run_id,
        "TRACK2_OUTPUT_MODEL_NAME": out_model,
        "TRACK2_FINAL_EVAL": "0",
        "NO_PROXY": no_proxy,
        "no_proxy": no_proxy,
        "PYTHONPATH": f"{CODEX}/wrappers:{CODEX}:" + env.get("PYTHONPATH", ""),
    })
    for key in ["HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "https_proxy", "http_proxy", "all_proxy"]:
        env.pop(key, None)
    return env


def run_spec(clean_dir: Path, item: Tuple[str, int, str, int], run_id: str, model: str, timeout: int) -> Dict[str, Any]:
    scenario, number, spec, task_count = item
    out_model = f"{model}-{RUN_VERSION}-{run_id}"
    env = make_env(run_id, model, out_model)
    log_dir = CODEX / "runs" / VERSION / run_id / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    clean_file = clean_dir / f"{spec}.json"
    official = EGO / "scenarios" / "final" / f"{spec}.json"
    backup = CODEX / "runs" / "V8_tmp_scenarios" / run_id / f"{spec}.official_backup.json"
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
        str(task_count),
    ]
    started = time.time()
    rc = 999
    timed_out = False
    error = ""
    log_path = log_dir / f"{spec}.log"
    try:
        shutil.copy2(official, backup)
        shutil.copy2(clean_file, official)
        with log_path.open("w", encoding="utf-8") as log:
            log.write(f"[v17 clean] spec={spec} tasks={task_count} run_id={run_id}\n")
            log.flush()
            try:
                cp = subprocess.run(cmd, cwd=str(EGO), env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
                rc = cp.returncode
            except subprocess.TimeoutExpired:
                timed_out = True
                rc = 124
                log.write(f"\n[v17 clean] timed out after {timeout}s\n")
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
        "number": number,
        "spec": spec,
        "task_count": task_count,
        "returncode": rc,
        "timed_out": timed_out,
        "error": error,
        "seconds": round(time.time() - started, 2),
        "output_model": out_model,
        "result_file": str(result_path),
        "log": str(log_path),
    }


def evaluate_subset(clean_dir: Path, run_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    sys.path.insert(0, str(EGO / "analysis_scripts"))
    sys.path.insert(0, str(EGO))
    from evaluate_interaction import evaluate_interaction_success
    import argparse as _argparse

    rows = []
    for item in sorted(run_items, key=lambda x: x["spec"]):
        gt_path = clean_dir / f"{item['spec']}.json"
        result = Path(item["result_file"])
        if not result.exists():
            rows.append({**item, "valid": 0, "joint": 0, "result": 0, "tool": 0, "micro": 0, "eval_error": "missing_result"})
            continue
        try:
            metrics = evaluate_interaction_success(
                str(gt_path),
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
    valid = sum(r.get("valid", 0) for r in rows)
    correct = sum(r.get("correct_calls", 0) for r in rows)
    gt_calls = sum(r.get("gt_calls", 0) for r in rows)

    def wavg(key: str) -> float:
        return sum(r.get(key, 0) * r.get("valid", 0) for r in rows) / valid if valid else 0.0

    return {
        "rows": rows,
        "summary": {
            "valid": valid,
            "joint": wavg("joint"),
            "result": wavg("result"),
            "tool": wavg("tool"),
            "micro": correct / gt_calls if gt_calls else wavg("micro"),
            "avg_task_accuracy": wavg("avg_task_accuracy"),
            "correct_calls": correct,
            "gt_calls": gt_calls,
            "interaction_calls": sum(r.get("interaction_calls", 0) for r in rows),
        },
    }


def write_report(run_id: str, clean_dir: Path, run_items: List[Dict[str, Any]], eval_result: Dict[str, Any]) -> Path:
    report = CODEX / "reports" / f"V17_CLEAN_EVAL_{run_id}.md"
    s = eval_result["summary"]
    lines = [
        f"# V17 Clean-Only Evaluation {run_id}",
        "",
        f"- version: {RUN_VERSION}",
        "- final_run: false",
        "- auto_submit: false",
        "- v10_zip_overwritten: false",
        "- uses_val41_gt_for_policy: false",
        "- uses_val41_gt_for_eval_only: true",
        "- uses_final_hidden_metadata_for_policy: false",
        f"- clean_split: `{clean_dir}`",
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
        "## Per Spec",
        "",
        "| spec | n | rc | valid | joint | result | tool | micro | calls | interaction_calls | error |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    by_spec = {x["spec"]: x for x in run_items}
    for row in sorted(eval_result["rows"], key=lambda x: x["spec"]):
        item = by_spec.get(row["spec"], {})
        err = row.get("eval_error") or item.get("error", "")
        lines.append(
            f"| {row['spec']} | {item.get('task_count', '')} | {item.get('returncode', '')} | {row.get('valid', 0)} | "
            f"{row.get('joint', 0):.3f} | {row.get('result', 0):.3f} | {row.get('tool', 0):.3f} | "
            f"{row.get('micro', 0):.3f} | {row.get('correct_calls', 0)}/{row.get('gt_calls', 0)} | "
            f"{row.get('interaction_calls', 0)} | {err} |"
        )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default=f"v17_clean_{time.strftime('%Y%m%d_%H%M%S')}")
    ap.add_argument("--clean-dir", default="")
    ap.add_argument("--model", default=os.environ.get("TRACK2_OPENAI_MODEL", "gpt-5.5"))
    ap.add_argument("--max-workers", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=1200)
    args = ap.parse_args()
    if not load_shell_env_file(CODEX / "state" / ".openai_env").get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY missing in state/.openai_env")
    clean_dir = Path(args.clean_dir) if args.clean_dir else latest_clean_split()
    specs = load_specs(clean_dir)
    out_dir = CODEX / "runs" / VERSION / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "manifest.json", {
        "run_id": args.run_id,
        "version": VERSION,
        "run_version": RUN_VERSION,
        "model": args.model,
        "clean_dir": str(clean_dir),
        "specs": specs,
        "final_run": False,
        "uses_val41_gt_for_policy": False,
    })
    run_items: List[Dict[str, Any]] = []
    with cf.ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as ex:
        futs = [ex.submit(run_spec, clean_dir, spec, args.run_id, args.model, args.timeout) for spec in specs]
        for fut in cf.as_completed(futs):
            item = fut.result()
            run_items.append(item)
            print(json.dumps({"finished": item["spec"], "rc": item["returncode"], "seconds": item["seconds"], "error": item["error"]}, ensure_ascii=False), flush=True)
            write_json(out_dir / "run_items.partial.json", sorted(run_items, key=lambda x: x["spec"]))
    run_items = sorted(run_items, key=lambda x: x["spec"])
    write_json(out_dir / "run_items.json", run_items)
    eval_result = evaluate_subset(clean_dir, run_items)
    write_json(out_dir / "eval_summary.json", eval_result)
    report = write_report(args.run_id, clean_dir, run_items, eval_result)
    state = {
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "run_id": args.run_id,
        "version": VERSION,
        "run_version": RUN_VERSION,
        "model": args.model,
        "clean_dir": str(clean_dir),
        "report": str(report),
        "summary": eval_result["summary"],
        "final_run": False,
        "protected_best_updated": False,
    }
    write_json(CODEX / "state" / "latest_v17_clean_eval.json", state)
    print(json.dumps(state, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
