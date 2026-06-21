# -*- coding: utf-8 -*-
"""Track2 unattended phase-1 autopilot.

This script intentionally keeps official EgoBench mostly untouched. It copies
entry scripts into codex/runners, injects wrappers with PYTHONPATH, runs
incremental smoke/baseline experiments, analyzes failures, and records status.
"""

import argparse
import csv
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time


EGO_ROOT = Path(os.environ.get("EGO_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/EgoBench"))
CODEX_ROOT = Path(os.environ.get("CODEX_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/codex"))
SCENARIOS_DEV = {
    "retail": [1, 2, 3, 4, 5, 7, 8, 9],
    "kitchen": [1, 2, 3],
    "restaurant": [1, 2, 3, 4],
    "order": [1],
}
SCENARIOS_FINAL = {"retail": [6, 10], "kitchen": [4], "restaurant": [5], "order": [2]}
PYTHON_BIN = os.environ.get("TRACK2_PYTHON") or shutil.which("python3") or shutil.which("python") or "python3"


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def load_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def append_jsonl(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def status_update(**kwargs):
    p = CODEX_ROOT / "state" / "autopilot_status.json"
    data = load_json(p, {})
    data.update(kwargs)
    data["updated_at"] = now()
    write_json(p, data)
    update_readme()


def update_readme():
    status = load_json(CODEX_ROOT / "state" / "autopilot_status.json", {})
    best = load_json(CODEX_ROOT / "state" / "best_version.json", {})
    text = [
        "# Track2 Codex Autopilot Status",
        "",
        f"- updated_at: {now()}",
        f"- ego_root: {EGO_ROOT}",
        f"- codex_root: {CODEX_ROOT}",
        f"- stage: {status.get('stage', 'unknown')}",
        f"- version: {status.get('version', 'unknown')}",
        f"- run_id: {status.get('run_id', 'unknown')}",
        f"- completed_tasks: {status.get('completed_tasks', 0)}",
        f"- best_version: {best.get('version', 'unknown')}",
        f"- best_joint_success: {best.get('joint_success', 'unknown')}",
        f"- final_handled: {status.get('final_handled', False)}",
        "",
        "## Launch",
        "",
        "```bash",
        f"nohup bash {CODEX_ROOT}/scripts/track2_autopilot.sh --resume > {CODEX_ROOT}/logs/autopilot.nohup.log 2>&1 &",
        f"nohup bash {CODEX_ROOT}/scripts/track2_watchdog.sh --resume > {CODEX_ROOT}/logs/watchdog.nohup.log 2>&1 &",
        "```",
        "",
        "If tmux is available:",
        "",
        "```bash",
        "tmux attach -t track2_autopilot",
        "tmux attach -t track2_watchdog",
        "```",
        "",
        "## Reproduce",
        "",
        f"Run scripts from `{CODEX_ROOT}`. Official EgoBench stays at `{EGO_ROOT}`.",
        "",
        "## Rollback",
        "",
        "Official files are not edited by default. If patches appear in `patches/`, restore matching backup from `backups/`.",
        "",
        "## Notes",
        "",
        "- Final submission is never auto-submitted.",
        "- API secrets are stored only in `state/secrets.env` and are not printed in reports.",
    ]
    with open(CODEX_ROOT / "README_STATUS.md", "w", encoding="utf-8") as f:
        f.write("\n".join(text) + "\n")


def run_command(cmd, cwd, log_file, env=None, timeout=None):
    start = time.time()
    env2 = os.environ.copy()
    env2.update(env or {})
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "a", encoding="utf-8") as log:
        log.write(f"\n[{now()}] CMD cwd={cwd} cmd={' '.join(cmd)}\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env2,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                log.write(line)
                log.flush()
            code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            code = -9
            log.write(f"\n[{now()}] TIMEOUT after {timeout}s\n")
    return {"cmd": cmd, "cwd": str(cwd), "exit_code": code, "elapsed": round(time.time() - start, 3), "log": str(log_file)}


def py_compile(paths):
    for path in paths:
        if path.exists():
            res = run_command([PYTHON_BIN, "-m", "py_compile", str(path)], CODEX_ROOT, CODEX_ROOT / "logs" / "compile.log")
            if res["exit_code"] != 0:
                raise RuntimeError(f"py_compile failed: {path}")


def bash_n(paths):
    for path in paths:
        if path.exists():
            res = run_command(["bash", "-n", str(path)], CODEX_ROOT, CODEX_ROOT / "logs" / "compile.log")
            if res["exit_code"] != 0:
                raise RuntimeError(f"bash -n failed: {path}")


def prepare_runners():
    status_update(stage="prepare_runners")
    runners = CODEX_ROOT / "runners"
    runners.mkdir(parents=True, exist_ok=True)
    src = EGO_ROOT / "run" / "multi_agent.py"
    dst = runners / "track2_multi_agent_plus.py"
    text = src.read_text(encoding="utf-8")
    if "egobench_agent_plus" not in text:
        text = text.replace(
            "from run.utils import (\n",
            "from run.utils import (\n",
        )
        text = text.replace(
            "from config.service_agent_config import get_video_path, VIDEO_MODE, SERVICE_MODEL_NAME\n",
            "from config.service_agent_config import get_video_path, VIDEO_MODE, SERVICE_MODEL_NAME\n"
            "from egobench_agent_plus.service_agent_wrapper import maybe_repair_agent_reply, enhance_service_prompt\n",
        )
        text = text.replace(
            "from egobench_agent_plus.service_agent_wrapper import maybe_repair_agent_reply, enhance_service_prompt\n",
            "from egobench_agent_plus.service_agent_wrapper import maybe_repair_agent_reply, enhance_service_prompt\n"
            "from egobench_agent_plus.direct_api import call_llm_direct\n"
            "call_llm = call_llm_direct\n",
        )
        text = text.replace(
            "service_agent_sys_prompt = SERVICE_AGENT_PROMPT_BASE.format(tool_descriptions=tool_descriptions)",
            "service_agent_sys_prompt = enhance_service_prompt(SERVICE_AGENT_PROMPT_BASE.format(tool_descriptions=tool_descriptions), args.scenario)",
        )
        text = text.replace(
            "service_agent_sys_prompt = enhance_service_prompt(SERVICE_AGENT_PROMPT_BASE.format(tool_descriptions=tool_descriptions), args.scenario)",
            "service_agent_sys_prompt = enhance_service_prompt(SERVICE_AGENT_PROMPT_BASE.format(tool_descriptions=tool_descriptions), args.scenario)\n"
            "        if os.environ.get(\"TRACK2_TEXT_ONLY_VISUAL_CONTEXT\", \"1\") == \"1\" and image_description:\n"
            "            service_agent_sys_prompt += \"\\n\\nVideo/action context description from benchmark metadata:\\n\" + image_description",
        )
        text = text.replace(
            "build_message_with_image(msg[\"content\"], image_path, use_vision=True, service_model_name=args.service_model_name)",
            "build_message_with_image(msg[\"content\"], image_path, use_vision=(os.environ.get(\"TRACK2_USE_VIDEO\", \"0\") == \"1\"), service_model_name=args.service_model_name)",
        )
        text = text.replace(
            "max_turns = 10",
            "max_turns = int(os.environ.get(\"TRACK2_MAX_TURNS\", \"10\"))",
        )
        text = text.replace(
            "print(f\"Tested Agent: {agent_reply}\")\n\n                    is_tool, tool_call_obj = check_tool_call(agent_reply)",
            "print(f\"Tested Agent: {agent_reply}\")\n"
            "                    agent_reply = maybe_repair_agent_reply(agent_reply, args.scenario, local_service_history)\n"
            "                    print(f\"Guarded Agent: {agent_reply}\")\n\n"
            "                    is_tool, tool_call_obj = check_tool_call(agent_reply)",
        )
        text = text.replace(
            'OUTPUT_JSON = f"./results/{args.service_model_name}/{args.scenario}{args.scenario_number}_easy.json"',
            'output_model_name = os.environ.get("TRACK2_OUTPUT_MODEL_NAME", args.service_model_name)\n'
            '    OUTPUT_JSON = f"./results/{output_model_name}/{args.scenario}{args.scenario_number}_easy.json"',
        )
    dst.write_text(text, encoding="utf-8")

    run_all = CODEX_ROOT / "runners" / "run_all_scenarios_plus.sh"
    run_all.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"CODEX_ROOT=\"${{CODEX_ROOT:-{CODEX_ROOT}}}\"\n"
        f"EGO_ROOT=\"${{EGO_ROOT:-{EGO_ROOT}}}\"\n"
        "source \"$CODEX_ROOT/scripts/track2_common.sh\"\n"
        "NUM_TASKS=0\n"
        "VERSION=${TRACK2_VERSION:-V1_format_schema_guard}\n"
        "while [[ $# -gt 0 ]]; do\n"
        "  case \"$1\" in\n"
        "    --num_tasks) NUM_TASKS=\"$2\"; shift 2 ;;\n"
        "    --version) VERSION=\"$2\"; shift 2 ;;\n"
        "    *) echo \"Unknown option: $1\"; exit 2 ;;\n"
        "  esac\n"
        "done\n"
        "PY=$(pick_python)\n"
        "cd \"$EGO_ROOT\"\n"
        "for spec in retail:1 retail:2 retail:3 retail:4 retail:5 retail:7 retail:8 retail:9 kitchen:1 kitchen:2 kitchen:3 restaurant:1 restaurant:2 restaurant:3 restaurant:4 order:1; do\n"
        "  scenario=\"${spec%%:*}\"; num=\"${spec##*:}\"\n"
        "  echo \"Running $VERSION $scenario$num num_tasks=$NUM_TASKS\"\n"
        "  \"$PY\" \"$CODEX_ROOT/runners/track2_multi_agent_plus.py\" --scenario \"$scenario\" --scenario_number \"$num\" --service_model_name \"$SERVICE_MODEL_NAME\" --multi_agent_user --summary_user --num_tasks \"$NUM_TASKS\" || true\n"
        "done\n",
        encoding="utf-8",
    )
    os.chmod(run_all, 0o755)
    py_compile([dst])
    bash_n([run_all])


def count_tasks_in_file(path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return len(data) if isinstance(data, list) else 0
    except Exception:
        return 0


def scenario_specs(limit_final=False):
    specs = []
    source = SCENARIOS_FINAL if limit_final else SCENARIOS_DEV
    for scenario, nums in source.items():
        for n in nums:
            p = EGO_ROOT / "scenarios" / "final" / f"{scenario}{n}.json"
            if p.exists():
                specs.append((scenario, n, count_tasks_in_file(p)))
    return specs


def run_one_scenario(version, run_id, scenario, number, num_tasks, use_plus):
    py = PYTHON_BIN
    runner = CODEX_ROOT / "runners" / "track2_multi_agent_plus.py" if use_plus else EGO_ROOT / "run" / "multi_agent.py"
    log = CODEX_ROOT / "runs" / version / run_id / "logs" / f"{scenario}{number}_{num_tasks}.log"
    output_model = os.environ.get("TRACK2_OUTPUT_MODEL_NAME", os.environ.get("SERVICE_MODEL_NAME", "deepseek-v4-pro"))
    env = {
        "PYTHONPATH": f"{CODEX_ROOT}:{EGO_ROOT}:{os.environ.get('PYTHONPATH', '')}",
        "TRACK2_VERSION": version,
        "TRACK2_OUTPUT_MODEL_NAME": output_model,
    }
    cmd = [
        py,
        str(runner),
        "--scenario",
        scenario,
        "--scenario_number",
        str(number),
        "--service_model_name",
        os.environ.get("SERVICE_MODEL_NAME", "deepseek-v4-pro"),
        "--multi_agent_user",
        "--summary_user",
        "--num_tasks",
        str(num_tasks),
    ]
    res = run_command(cmd, EGO_ROOT, log, env=env, timeout=3600)
    append_jsonl(CODEX_ROOT / "state" / "commands.jsonl", {
        "time": now(),
        "version": version,
        "run_id": run_id,
        "scenario": scenario,
        "number": number,
        "num_tasks": num_tasks,
        **res,
    })
    if res["exit_code"] != 0:
        err_type = classify_run_error(log)
        append_jsonl(CODEX_ROOT / "state" / "error_queue.jsonl", {
            "time": now(),
            "stage": "run_scenario",
            "version": version,
            "scenario": scenario,
            "number": number,
            "exit_code": res["exit_code"],
            "error_type": err_type,
            "log": str(log),
        })
    return res


def classify_run_error(log_path):
    try:
        text = Path(log_path).read_text(encoding="utf-8", errors="replace")[-6000:]
    except Exception:
        return "unknown"
    if "ModuleNotFoundError" in text or "ImportError" in text:
        return "import_error"
    if "NameError: name 'OpenAI'" in text:
        return "openai_import_error"
    if "Error code: 401" in text or "Unauthorized" in text or "invalid api" in text.lower():
        return "api_auth_error"
    if "Connection" in text or "timeout" in text.lower():
        return "api_connection_error"
    if "No such file" in text or "FileNotFoundError" in text:
        return "path_error"
    return "runtime_error"


def run_eval(model_name, run_id, version, num_samples):
    log = CODEX_ROOT / "runs" / version / run_id / "logs" / f"eval_{num_samples}.log"
    bin_dir = CODEX_ROOT / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    python_link = bin_dir / "python"
    try:
        if not python_link.exists():
            python_link.symlink_to(PYTHON_BIN)
    except Exception:
        pass
    cmd = ["bash", str(EGO_ROOT / "analysis_scripts" / "run_eval.sh"), "--model_name", model_name, "--num_samples", str(num_samples)]
    env = {"PATH": f"{bin_dir}:{os.environ.get('PATH', '')}"}
    res = run_command(cmd, EGO_ROOT, log, env=env, timeout=1800)
    append_jsonl(CODEX_ROOT / "state" / "commands.jsonl", {"time": now(), "stage": "eval", "version": version, "run_id": run_id, **res})
    return res


def collect_eval_summary(model_name):
    summary_path = EGO_ROOT / "eval_result" / model_name / "summary.json"
    data = load_json(summary_path, {})
    summary = data.get("summary", data) if isinstance(data, dict) else {}

    def metric(*names):
        for name in names:
            if isinstance(summary, dict) and name in summary:
                return summary.get(name, 0)
            if isinstance(data, dict) and name in data:
                return data.get(name, 0)
        return 0

    return {
        "avg_tool_based_success_rate": metric("avg_tool_based_success_rate", "tool_success_rate"),
        "avg_result_based_success_rate": metric("avg_result_based_success_rate", "result_success_rate"),
        "avg_joint_success_rate": metric("avg_joint_success_rate", "joint_success_rate"),
        "micro_accuracy": metric("micro_accuracy", "micro_tool_accuracy"),
        "summary_path": str(summary_path),
    }


def copy_results(version, run_id, model_name):
    dest = CODEX_ROOT / "runs" / version / run_id
    dest.mkdir(parents=True, exist_ok=True)
    for sub in ["results", "eval_result"]:
        src = EGO_ROOT / sub / model_name
        if src.exists():
            dst = dest / sub / model_name
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)


def parse_gate_specs():
    raw = os.environ.get("TRACK2_GATE_SPECS", "retail:9,kitchen:2,restaurant:4,order:1")
    specs = []
    for part in raw.split(","):
        part = part.strip()
        if not part or ":" not in part:
            continue
        scenario, num = part.split(":", 1)
        try:
            specs.append((scenario.strip(), int(num)))
        except ValueError:
            continue
    return specs


def run_gate_matrix(run_id):
    version = "V2_prompt_plan"
    status_update(stage="gate_matrix", version=version, run_id=run_id)
    os.environ.setdefault("TRACK2_OUTPUT_MODEL_NAME", f"deepseek-v4-pro-gate-{run_id}")
    os.environ.setdefault("TRACK2_ENABLE_PLANNER", "1")
    os.environ.setdefault("TRACK2_ENABLE_SCENARIO_RULES", "1")
    os.environ.setdefault("TRACK2_API_MAX_RETRIES", "3")
    os.environ.setdefault("TRACK2_MAX_TURNS", "8")
    os.environ.setdefault("TRACK2_CONNECT_TIMEOUT", "8")
    os.environ.setdefault("TRACK2_READ_TIMEOUT", "160")
    model = os.environ.get("SERVICE_MODEL_NAME", "deepseek-v4-pro")
    output_model = os.environ["TRACK2_OUTPUT_MODEL_NAME"]
    run_dir = CODEX_ROOT / "runs" / version / run_id
    pre_dir = run_dir / "preexisting"
    pre_dir.mkdir(parents=True, exist_ok=True)

    for scenario, number in parse_gate_specs():
        input_path = EGO_ROOT / "scenarios" / "final" / f"{scenario}{number}.json"
        if not input_path.exists():
            append_jsonl(CODEX_ROOT / "state" / "error_queue.jsonl", {
                "time": now(),
                "stage": "gate_matrix",
                "scenario": scenario,
                "number": number,
                "error_type": "missing_scenario",
                "path": str(input_path),
            })
            continue
        out = EGO_ROOT / "results" / output_model / f"{scenario}{number}_easy.json"
        if out.exists():
            shutil.copy2(out, pre_dir / f"{scenario}{number}_easy.before.json")
        res = run_one_scenario(version, run_id, scenario, number, 1, use_plus=True)
        if res["exit_code"] != 0:
            append_jsonl(CODEX_ROOT / "state" / "error_queue.jsonl", {
                "time": now(),
                "stage": "gate_matrix",
                "scenario": scenario,
                "number": number,
                "exit_code": res["exit_code"],
                "error_type": classify_run_error(res["log"]),
                "log": res["log"],
            })
        status = load_json(CODEX_ROOT / "state" / "autopilot_status.json", {})
        status_update(completed_tasks=int(status.get("completed_tasks", 0)) + 1)

    run_eval(output_model, run_id, version, 1)
    summary = collect_eval_summary(output_model)
    copy_results(version, run_id, output_model)
    analyze_failures(run_id)
    experiment_matrix(run_id, {version: summary})
    write_json(CODEX_ROOT / "state" / "best_version.json", {
        "version": version,
        "output_model_name": output_model,
        "service_model_name": model,
        "joint_success": summary.get("avg_joint_success_rate", 0),
        "result_success": summary.get("avg_result_based_success_rate", 0),
        "micro_tool_accuracy": summary.get("micro_accuracy", 0),
        "summary_path": summary.get("summary_path", ""),
        "updated_at": now(),
    })
    status_update(stage="gate_complete", version=version, run_id=run_id)
    return summary


def write_baseline_report(run_id, version, num_tasks, summary):
    report = CODEX_ROOT / "reports" / f"01_baseline_summary_{run_id}.md"
    lines = [
        f"# Baseline Summary {run_id}",
        "",
        f"- version: {version}",
        f"- num_tasks_per_scenario: {num_tasks}",
        f"- environment_model: {os.environ.get('SERVICE_MODEL_NAME', '')}",
        f"- tool_success: {summary.get('avg_tool_based_success_rate', 0)}",
        f"- result_success: {summary.get('avg_result_based_success_rate', 0)}",
        f"- joint_success: {summary.get('avg_joint_success_rate', 0)}",
        f"- micro_tool_accuracy: {summary.get('micro_accuracy', 0)}",
        f"- summary_path: {summary.get('summary_path', '')}",
        "",
        "## Bottleneck",
        "",
        "See failure analysis reports in `analysis/` and `reports/02_failure_analysis_*`.",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_version(version, run_id, num_tasks, use_plus):
    status_update(stage="run_version", version=version, run_id=run_id)
    model = os.environ.get("SERVICE_MODEL_NAME", "deepseek-v4-pro")
    specs = scenario_specs(limit_final=False)
    completed = 0
    for scenario, number, total in specs:
        if total <= 0:
            continue
        out = EGO_ROOT / "results" / model / f"{scenario}{number}_easy.json"
        if out.exists() and len(load_json(out, [])) >= min(num_tasks, total):
            completed += min(num_tasks, total)
            status_update(completed_tasks=completed)
            continue
        res = run_one_scenario(version, run_id, scenario, number, num_tasks, use_plus)
        if res["exit_code"] != 0:
            err_type = classify_run_error(res["log"])
            if err_type in {"import_error", "openai_import_error", "api_auth_error"}:
                status_update(stage="NEED_HUMAN_ATTENTION" if err_type == "api_auth_error" else "blocked_environment", version=version, run_id=run_id)
                raise RuntimeError(f"Environment/API blocker during {version} {scenario}{number}: {err_type}. See {res['log']}")
        completed += min(num_tasks, total)
        status_update(completed_tasks=completed)
    run_eval(model, run_id, version, num_tasks)
    summary = collect_eval_summary(model)
    copy_results(version, run_id, model)
    if version == "V0_official_baseline":
        write_baseline_report(run_id, version, num_tasks, summary)
    return summary


def analyze_failures(run_id):
    status_update(stage="failure_analysis", run_id=run_id)
    py = PYTHON_BIN
    log = CODEX_ROOT / "logs" / f"failure_analysis_{run_id}.log"
    return run_command([py, str(CODEX_ROOT / "scripts" / "track2_analyze_failures.py"), "--run-id", run_id, "--resume"], CODEX_ROOT, log, timeout=1800)


def make_visual_cache(run_id, dry_run=False):
    status_update(stage="visual_cache", run_id=run_id)
    py = PYTHON_BIN
    args = [py, str(CODEX_ROOT / "scripts" / "track2_make_visual_cache.py"), "--run-id", run_id, "--resume"]
    if dry_run:
        args.append("--dry-run")
    log = CODEX_ROOT / "logs" / f"visual_cache_{run_id}.log"
    return run_command(args, CODEX_ROOT, log, timeout=1800)


def pack_final_if_present(run_id):
    present = all((EGO_ROOT / rel).exists() for rel in [
        "scenarios/final/retail6.json",
        "scenarios/final/retail10.json",
        "scenarios/final/kitchen4.json",
        "scenarios/final/restaurant5.json",
        "scenarios/final/order2.json",
    ])
    if not present:
        status_update(final_handled=False)
        return
    py = PYTHON_BIN
    log = CODEX_ROOT / "logs" / f"pack_submission_{run_id}.log"
    run_command([py, str(CODEX_ROOT / "scripts" / "track2_pack_submission.py"), "--run-id", run_id, "--resume"], CODEX_ROOT, log, timeout=1800)
    status_update(final_handled=True)


def experiment_matrix(run_id, summaries):
    status_update(stage="experiment_matrix", run_id=run_id)
    matrix_csv = CODEX_ROOT / "analysis" / f"experiment_matrix_{run_id}.csv"
    matrix_md = CODEX_ROOT / "reports" / f"03_experiment_matrix_{run_id}.md"
    rows = []
    for version, summary in summaries.items():
        rows.append({
            "version": version,
            "scenario": "all_dev",
            "mode": "easy",
            "num_tasks": 20,
            "micro_tool_accuracy": summary.get("micro_accuracy", 0),
            "tool_success": summary.get("avg_tool_based_success_rate", 0),
            "result_success": summary.get("avg_result_based_success_rate", 0),
            "joint_success": summary.get("avg_joint_success_rate", 0),
            "avg_turns": "",
            "avg_tool_calls": "",
            "invalid_json_rate": "",
            "invalid_tool_rate": "",
            "missing_param_rate": "",
            "risky_op_rate": "",
            "repeated_loop_rate": "",
            "runtime": "",
            "notes": summary.get("summary_path", ""),
        })
    matrix_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(matrix_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["version"])
        writer.writeheader()
        writer.writerows(rows)
    best = None
    for row in rows:
        key = (
            float(row.get("joint_success") or 0),
            float(row.get("result_success") or 0),
            float(row.get("micro_tool_accuracy") or 0),
        )
        if best is None or key > best[0]:
            best = (key, row)
    if best:
        write_json(CODEX_ROOT / "state" / "best_version.json", {
            "version": best[1]["version"],
            "joint_success": best[1]["joint_success"],
            "result_success": best[1]["result_success"],
            "micro_tool_accuracy": best[1]["micro_tool_accuracy"],
            "updated_at": now(),
        })
    lines = ["# Experiment Matrix", "", f"- run_id: {run_id}", f"- csv: {matrix_csv}", ""]
    for row in rows:
        lines.append(f"- {row['version']}: joint={row['joint_success']} result={row['result_success']} micro={row['micro_tool_accuracy']}")
    if best:
        lines.append("")
        lines.append(f"Best version: {best[1]['version']}")
    matrix_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    update_readme()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=f"track2_{time.strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--full-matrix", action="store_true")
    args = parser.parse_args()

    CODEX_ROOT.mkdir(parents=True, exist_ok=True)
    status_update(stage="start", run_id=args.run_id)

    if args.dry_run:
        prepare_runners()
        status_update(stage="dry_run_complete", run_id=args.run_id)
        return

    prepare_runners()

    if not args.full_matrix:
        run_gate_matrix(args.run_id)
        pack_final_if_present(args.run_id)
        return

    summaries = {}

    # Step 4-5: official smoke and V0 small baseline.
    summaries["V0_official_baseline"] = run_version("V0_official_baseline", args.run_id, 1, use_plus=False)
    analyze_failures(args.run_id)

    # Step 7-8: V1 guard, 20 tasks per scenario.
    summaries["V1_format_schema_guard"] = run_version("V1_format_schema_guard", args.run_id, 20, use_plus=True)
    analyze_failures(args.run_id)

    # Step 9-13: enable same copied runner with stronger prompt flags.
    os.environ["TRACK2_ENABLE_PLANNER"] = "1"
    summaries["V2_prompt_plan"] = run_version("V2_prompt_plan", args.run_id, 20, use_plus=True)
    os.environ["TRACK2_ENABLE_DB_GUARD"] = "1"
    summaries["V3_db_guard"] = run_version("V3_db_guard", args.run_id, 20, use_plus=True)
    make_visual_cache(args.run_id)
    os.environ["TRACK2_ENABLE_VISUAL_CACHE"] = "1"
    summaries["V4_visual_cache"] = run_version("V4_visual_cache", args.run_id, 20, use_plus=True)
    os.environ["TRACK2_ENABLE_SCENARIO_RULES"] = "1"
    summaries["V5_full"] = run_version("V5_full", args.run_id, 20, use_plus=True)
    experiment_matrix(args.run_id, summaries)
    pack_final_if_present(args.run_id)
    status_update(stage="complete", run_id=args.run_id)


if __name__ == "__main__":
    main()
