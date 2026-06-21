# -*- coding: utf-8 -*-
"""Analyze EgoBench Track2 failures from results and eval_result."""

import argparse
import csv
import json
import os
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List


EGO_ROOT = Path(os.environ.get("EGO_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/EgoBench"))
CODEX_ROOT = Path(os.environ.get("CODEX_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/codex"))


FIELDS = [
    "task_id", "scenario", "mode", "success_joint", "success_tool", "success_result",
    "micro_tool_acc", "turns", "num_tool_calls", "invalid_json_count",
    "invalid_tool_name_count", "missing_required_param_count", "wrong_param_type_count",
    "mixed_natural_language_and_json_count", "premature_stop", "timeout_or_no_stop",
    "repeated_tool_loop", "retrieval_missing_before_modify", "risky_state_modification",
    "suspected_visual_error", "suspected_logic_error", "suspected_hallucination",
    "final_db_mismatch", "first_bad_turn", "raw_agent_output_at_first_bad_turn",
    "suggested_fix", "file",
]


TOOL_JSON_RE = re.compile(r"^\s*(\[|\{)")
NL_AND_JSON_RE = re.compile(r"\[[\s\S]*tool_name[\s\S]*\].+\S|.+\S[\s\S]*\[[\s\S]*tool_name", re.IGNORECASE)
STATE_CHANGE_RE = re.compile(r"(add|remove|delete|update|modify|clear|set|create|place|cancel|cart|order|list|menu)", re.IGNORECASE)
RETRIEVAL_RE = re.compile(r"(get|find|query|search|check|calculate|compute|list)", re.IGNORECASE)
VISUAL_HINT_RE = re.compile(r"(left|right|front|back|color|red|green|blue|yellow|black|white|point|hold|second|first|visible|video)", re.IGNORECASE)
LOGIC_HINT_RE = re.compile(r"(cheapest|highest|lowest|same|origin|budget|less|more|expired|nutrition|calorie|protein|discount|tax|if|when)", re.IGNORECASE)


def load_json(path: Path, default: Any) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def iter_result_files(model_name: str = "") -> Iterable[Path]:
    root = EGO_ROOT / "results"
    if model_name:
        yield from sorted((root / model_name).glob("*.json"))
    else:
        yield from sorted(root.glob("*/*.json"))


def parse_name(path: Path):
    stem = path.stem
    mode = "easy" if stem.endswith("_easy") else ""
    base = stem[:-5] if stem.endswith("_easy") else stem
    m = re.match(r"([a-zA-Z]+)(\d+)", base)
    if not m:
        return base, "", mode
    return m.group(1), m.group(2), mode


def eval_lookup(model: str, scenario: str, number: str, task_id: int) -> Dict[str, Any]:
    path = EGO_ROOT / "eval_result" / model / f"{scenario}{number}_easy_eval.json"
    data = load_json(path, {})
    for d in data.get("detailed_results", []):
        if int(d.get("task_id", -1)) == int(task_id):
            return d
    return {}


def flatten_tool_calls(task: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for group in task.get("tool_calls", []):
        for call in group.get("calls", []) if isinstance(group, dict) else []:
            out.append(call)
    return out


def count_invalid_agent_outputs(task: Dict[str, Any]) -> Dict[str, Any]:
    invalid_json = 0
    mixed = 0
    first_bad_turn = ""
    first_bad = ""
    for item in task.get("dialogue", []):
        if item.get("role") != "agent":
            continue
        text = str(item.get("content", ""))
        if "tool_name" in text and not TOOL_JSON_RE.search(text):
            invalid_json += 1
            if first_bad == "":
                first_bad_turn = str(item.get("turn", ""))
                first_bad = text
        if NL_AND_JSON_RE.search(text):
            mixed += 1
            if first_bad == "":
                first_bad_turn = str(item.get("turn", ""))
                first_bad = text
    return {"invalid_json_count": invalid_json, "mixed": mixed, "first_bad_turn": first_bad_turn, "first_bad": first_bad}


def detect_loop(calls: List[Dict[str, Any]]) -> bool:
    seen = {}
    for call in calls:
        key = json.dumps(call, ensure_ascii=False, sort_keys=True)
        seen[key] = seen.get(key, 0) + 1
        if seen[key] >= 3:
            return True
    return False


def retrieval_before_modify(calls: List[Dict[str, Any]]) -> bool:
    saw_retrieval = False
    for call in calls:
        name = str(call.get("tool_name", call.get("name", "")))
        if RETRIEVAL_RE.search(name):
            saw_retrieval = True
        if STATE_CHANGE_RE.search(name) and not saw_retrieval:
            return True
    return False


def classify(task: Dict[str, Any], detail: Dict[str, Any], calls: List[Dict[str, Any]], scenario_text: str) -> Dict[str, Any]:
    text = json.dumps(task, ensure_ascii=False)[:20000] + " " + scenario_text
    success_tool = bool(detail.get("tool_based", {}).get("success", False))
    success_result = bool(detail.get("result_based", {}).get("success", False))
    invalid = count_invalid_agent_outputs(task)
    repeated = detect_loop(calls)
    retrieval_missing = retrieval_before_modify(calls)
    risky = retrieval_missing or any("user_id" in c.get("parameters", {}) and not c.get("parameters", {}).get("user_id") for c in calls)
    visual = (not success_tool) and bool(VISUAL_HINT_RE.search(text))
    logic = (not success_tool or not success_result) and bool(LOGIC_HINT_RE.search(text))
    hallucination = (not success_tool) and any(word in text.lower() for word in ["not found", "error", "unknown", "fabricat"])
    suggested = []
    if invalid["invalid_json_count"] or invalid["mixed"]:
        suggested.append("Structural Non-Compliance: enforce pure JSON array or pure natural language.")
    if visual:
        suggested.append("Multimodal Perceptual Misinterpretation: use visual cache or ask targeted clarification.")
    if hallucination:
        suggested.append("Hallucination: trust tool results and avoid invented product/menu attributes.")
    if logic:
        suggested.append("Logical Fallacy: retrieve candidates and compare attributes before state change.")
    if risky:
        suggested.append("Risky Operation: retrieve before modifying DB and verify user_id/object.")
    if not suggested:
        suggested.append("Inspect first failed tool parameters and final DB hash mismatch.")
    return {
        "invalid_json_count": invalid["invalid_json_count"],
        "mixed_natural_language_and_json_count": invalid["mixed"],
        "first_bad_turn": invalid["first_bad_turn"],
        "raw_agent_output_at_first_bad_turn": invalid["first_bad"].replace("\n", " ")[:1000],
        "repeated_tool_loop": repeated,
        "retrieval_missing_before_modify": retrieval_missing,
        "risky_state_modification": risky,
        "suspected_visual_error": visual,
        "suspected_logic_error": logic,
        "suspected_hallucination": hallucination,
        "suggested_fix": " ".join(suggested),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="")
    parser.add_argument("--model-name", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_id = args.run_id or "latest"
    rows = []
    json_rows = []
    for result_path in iter_result_files(args.model_name):
        model = result_path.parent.name
        scenario, number, mode = parse_name(result_path)
        tasks = load_json(result_path, [])
        scenario_file = EGO_ROOT / "scenarios" / "final" / f"{scenario}{number}.json"
        scenario_gt = load_json(scenario_file, [])
        for idx, task in enumerate(tasks):
            task_id = int(task.get("task_id", idx + 1))
            detail = eval_lookup(model, scenario, number, task_id)
            calls = flatten_tool_calls(task)
            tool_total = detail.get("tool_based", {}).get("total_gt_calls", 0)
            tool_matches = detail.get("tool_based", {}).get("matches", 0)
            scenario_text = json.dumps(scenario_gt[idx], ensure_ascii=False) if idx < len(scenario_gt) else ""
            flags = classify(task, detail, calls, scenario_text)
            row = {
                "task_id": task_id,
                "scenario": f"{scenario}{number}",
                "mode": mode,
                "success_joint": bool(detail.get("joint_success", False)),
                "success_tool": bool(detail.get("tool_based", {}).get("success", False)),
                "success_result": bool(detail.get("result_based", {}).get("success", False)),
                "micro_tool_acc": (tool_matches / tool_total) if tool_total else 0,
                "turns": task.get("rounds_count", 0),
                "num_tool_calls": task.get("tool_calls_count", len(calls)),
                "invalid_tool_name_count": "",
                "missing_required_param_count": "",
                "wrong_param_type_count": "",
                "premature_stop": task.get("rounds_count", 0) <= 1 and not detail.get("joint_success", False),
                "timeout_or_no_stop": task.get("rounds_count", 0) >= 10 and not detail.get("joint_success", False),
                "final_db_mismatch": bool(detail.get("tool_based", {}).get("success", False)) and not bool(detail.get("result_based", {}).get("success", False)),
                "file": str(result_path),
            }
            row.update(flags)
            rows.append(row)
            json_rows.append({"row": row, "eval_detail": detail, "tool_calls": calls})

    out_csv = CODEX_ROOT / "analysis" / f"failure_analysis_{run_id}.csv"
    out_json = CODEX_ROOT / "analysis" / f"failure_analysis_{run_id}.json"
    report = CODEX_ROOT / "reports" / f"02_failure_analysis_{run_id}.md"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        print(f"would_write rows={len(rows)}")
        return
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            clean = {k: str(row.get(k, "")).replace("\n", " ").replace("|", "/") for k in FIELDS}
            writer.writerow(clean)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(json_rows, f, ensure_ascii=False, indent=2)

    total = len(rows)
    joint_fail = sum(1 for r in rows if not r.get("success_joint"))
    structural = sum(1 for r in rows if int(r.get("invalid_json_count") or 0) > 0 or int(r.get("mixed_natural_language_and_json_count") or 0) > 0)
    risky = sum(1 for r in rows if r.get("risky_state_modification"))
    logic = sum(1 for r in rows if r.get("suspected_logic_error"))
    visual = sum(1 for r in rows if r.get("suspected_visual_error"))
    lines = [
        f"# Failure Analysis {run_id}",
        "",
        f"- total_tasks: {total}",
        f"- joint_failures: {joint_fail}",
        f"- structural_non_compliance: {structural}",
        f"- risky_operation: {risky}",
        f"- suspected_logic_error: {logic}",
        f"- suspected_visual_error: {visual}",
        f"- csv: {out_csv}",
        f"- json: {out_json}",
        "",
        "## Top Suggested Fixes",
        "",
        "1. Enforce tool-call JSON shape and schema validation.",
        "2. Retrieve before state-changing calls.",
        "3. Add compact planning hints for conditional filtering and candidate comparison.",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_csv}")
    print(f"wrote {out_json}")
    print(f"wrote {report}")


if __name__ == "__main__":
    main()
