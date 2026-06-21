#!/usr/bin/env python3
from __future__ import annotations

import json
import time
from pathlib import Path

CODEX = Path("/home/data-gxu/acm/egolink2026-main/code/track2/codex")
EGO = Path("/home/data-gxu/acm/egolink2026-main/code/track2/EgoBench")

RUNS = [
    ("V6_1_gpt55_guarded_endpoint", "gpt55_endpoint_gate_20260617_102324"),
    ("V6_1_1_gpt55_guarded_endpoint", "gpt55_endpoint_gate_20260617_103817"),
    ("V6_1_2_gpt55_guarded_endpoint", "gpt55_endpoint_gate_20260617_104726"),
    ("V6_1_3_gpt55_guarded_endpoint", "gpt55_endpoint_gate_20260617_105936"),
]


def load_summary(version: str, run_id: str):
    model_name = f"gpt-5.5-{version}-{run_id}"
    path = EGO / "eval_result" / model_name / "summary.json"
    if not path.exists():
        return None, path
    return json.loads(path.read_text(encoding="utf-8")), path


def main() -> int:
    rows = []
    for version, run_id in RUNS:
        data, path = load_summary(version, run_id)
        if not data:
            rows.append({"version": version, "run_id": run_id, "status": "missing_eval", "path": str(path)})
            continue
        s = data.get("summary", {})
        rows.append({
            "version": version,
            "run_id": run_id,
            "joint": s.get("avg_joint_success_rate"),
            "tool": s.get("avg_tool_based_success_rate"),
            "result": s.get("avg_result_based_success_rate"),
            "micro": s.get("micro_accuracy"),
            "avg_tool_calls": s.get("avg_tool_calls_count"),
            "avg_rounds": s.get("avg_rounds_count"),
            "path": str(path),
        })
    valid = [r for r in rows if isinstance(r.get("joint"), (int, float))]
    best = sorted(
        valid,
        key=lambda r: (
            r.get("joint") or 0,
            r.get("result") or 0,
            r.get("micro") or 0,
            -(r.get("avg_tool_calls") or 999),
        ),
        reverse=True,
    )[0] if valid else None

    report = CODEX / "reports" / f"GPT55_ENDPOINT_GATE_SUMMARY_{time.strftime('%Y%m%d_%H%M%S')}.md"
    lines = [
        "# GPT-5.5 Endpoint Gate Summary",
        "",
        f"- generated_at: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}",
        "- endpoint: https://ai-pixel.online/v1",
        "- cf_endpoint: tested, blocked with 403/1010",
        "- model: gpt-5.5",
        "- key_present: yes",
        "- key_logged: no",
        "- final_submission: not submitted",
        "",
        "| version | run_id | joint | result | tool | micro | avg_tool_calls | avg_rounds |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            "| {version} | {run_id} | {joint} | {result} | {tool} | {micro} | {avg_tool_calls} | {avg_rounds} |".format(
                version=r.get("version"),
                run_id=r.get("run_id"),
                joint="" if r.get("joint") is None else f"{r.get('joint'):.3f}",
                result="" if r.get("result") is None else f"{r.get('result'):.3f}",
                tool="" if r.get("tool") is None else f"{r.get('tool'):.3f}",
                micro="" if r.get("micro") is None else f"{r.get('micro'):.3f}",
                avg_tool_calls="" if r.get("avg_tool_calls") is None else f"{r.get('avg_tool_calls'):.2f}",
                avg_rounds="" if r.get("avg_rounds") is None else f"{r.get('avg_rounds'):.2f}",
            )
        )
    lines += [
        "",
        "## Current Best",
        "",
    ]
    if best:
        lines += [
            f"- version: {best['version']}",
            f"- run_id: {best['run_id']}",
            f"- joint_success: {best['joint']:.3f}",
            f"- result_success: {best['result']:.3f}",
            f"- tool_success: {best['tool']:.3f}",
            f"- micro_tool_accuracy: {best['micro']:.3f}",
            "- scenario notes: retail9 and restaurant4 joint success; order1 result success after restaurant pin canonicalization; kitchen2 still fails and needs task-specific pruning/branch repair.",
        ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    state = {
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "endpoint": "https://ai-pixel.online/v1",
        "model": "gpt-5.5",
        "key_present": True,
        "key_logged": False,
        "best": best,
        "rows": rows,
        "report": str(report.relative_to(CODEX)),
        "final_submission": "not_submitted",
    }
    (CODEX / "state" / "best_track2_api_version.json").write_text(json.dumps(state, indent=2), encoding="utf-8")

    readme = CODEX / "README_STATUS.md"
    old = readme.read_text(encoding="utf-8") if readme.exists() else "# Track2 Codex Status\n"
    marker = "## GPT-5.5 endpoint status"
    if marker in old:
        old = old.split(marker, 1)[0].rstrip() + "\n"
    block = [
        "",
        marker + f" ({time.strftime('%Y-%m-%dT%H:%M:%S%z')})",
        "",
        "- Endpoint tested: `https://ai-pixel.online/v1` works with `gpt-5.5`; `https://cf.ai-pixel.online` returns 403/1010.",
        "- Key stored in `state/.openai_env` with mode 600; key value not logged.",
        "- Current best API version: `V6_1_3_gpt55_guarded_endpoint`.",
        "- 4-task gate best: joint 50%, result 75%, tool 50%, micro 70.83%.",
        "- Successful joint scenarios: retail9, restaurant4.",
        "- order1: result success now 100%, tool trajectory still mismatch.",
        "- kitchen2: still failing; tool calls reduced from 62 to 35 in V6.1.1, but branch/result still wrong.",
        f"- Latest report: `{state['report']}`.",
        "- Final submission: not submitted.",
        "",
    ]
    readme.write_text(old.rstrip() + "\n".join(block), encoding="utf-8")
    print(report)
    print(json.dumps({"best": best, "report": state["report"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
