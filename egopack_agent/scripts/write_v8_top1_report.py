#!/usr/bin/env python3
from __future__ import annotations

import json
import time
from pathlib import Path

CODEX = Path("/home/data-gxu/acm/egolink2026-main/code/track2/codex")


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def summary_row(label: str, version: str, run_id: str, stage: str) -> dict:
    path = CODEX / "runs" / version / run_id / "eval_summary.json"
    data = load_json(path) or {"summary": {}, "rows": []}
    s = data.get("summary", {})
    return {
        "label": label,
        "version": version,
        "run_id": run_id,
        "stage": stage,
        "valid": s.get("valid", 0),
        "joint": s.get("joint", 0),
        "result": s.get("result", 0),
        "tool": s.get("tool", 0),
        "micro": s.get("micro", 0),
        "avg_task_accuracy": s.get("avg_task_accuracy", 0),
        "correct_calls": s.get("correct_calls", 0),
        "gt_calls": s.get("gt_calls", 0),
        "interaction_calls": s.get("interaction_calls", 0),
        "rows": data.get("rows", []),
    }


def fmt(x):
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


ts = time.strftime("%Y%m%d_%H%M%S")
rows = [
    summary_row("V8_6 all helpers smoke", "V8_6_top1_candidate", "v8_smoke_20260617_1823", "smoke"),
    summary_row("V8_0 stable smoke", "V8_0_v6_stable_reval", "v8_0_smoke_20260617_1832", "smoke"),
    summary_row("V8_1 order helper smoke", "V8_1_order_helper", "v8_1_smoke_20260617_continue", "smoke"),
    summary_row("V8_2 kitchen helper smoke", "V8_2_kitchen_helper", "v8_2_smoke_20260617_continue", "smoke"),
    summary_row("V8_0 stable validation_A_small", "V8_0_v6_stable_reval", "v8_0_valA_small_20260617_continue", "validation_A_small"),
    summary_row("V8_2 kitchen validation_A_small", "V8_2_kitchen_helper", "v8_2_valA_small_20260617_continue", "validation_A_small"),
]

best = load_json(CODEX / "state" / "best_track2_api_version.json") or {}

report = CODEX / "reports" / f"V8_TOP1_READINESS_{ts}.md"
lines = [
    f"# V8_TOP1_READINESS {ts}",
    "",
    "## Current Decision",
    "",
    "- Do not update `state/best_track2_api_version.json`.",
    "- Protected best remains `V6_1_3_gpt55_guarded_endpoint` from `gpt55_endpoint_gate_20260617_105936`.",
    "- Do not run final full inference or package submission from V8 yet.",
    "- No final submission was made.",
    "",
    "## Protected Best",
    "",
    f"- version: `{best.get('version','')}`",
    f"- run_id: `{best.get('run_id','')}`",
    f"- 4-task gate joint/result/tool/micro: {best.get('joint_success',0):.4f} / {best.get('result_success',0):.4f} / {best.get('tool_success',0):.4f} / {best.get('micro_tool_accuracy',0):.4f}",
    f"- endpoint/model: `{best.get('endpoint','')}` / `{best.get('model','')}`",
    "",
    "## Metric Fix",
    "",
    "- Fixed `scripts/run_v8_validation.py` micro extraction.",
    "- Official evaluator stores micro under `micro_tool_stats.micro_accuracy`, not `tool_based.micro_accuracy`.",
    "- Recomputed existing V8 smoke summaries without model/API calls.",
    "",
    "## V8 Results",
    "",
    "| label | stage | valid | joint | result | tool | micro | calls | interaction_calls |",
    "|---|---|---:|---:|---:|---:|---:|---:|---:|",
]

for r in rows:
    lines.append(
        f"| {r['label']} | {r['stage']} | {r['valid']} | {r['joint']:.4f} | {r['result']:.4f} | {r['tool']:.4f} | {r['micro']:.4f} | {r['correct_calls']}/{r['gt_calls']} | {r['interaction_calls']} |"
    )

lines += [
    "",
    "## Interpretation",
    "",
    "- `V8_2_kitchen_helper` is useful on the fixed 4-task smoke: 75% joint and 81.82% micro, with retail9/restaurant4/order1 all joint-success.",
    "- The same version does not generalize on the frozen 20-task `validation_A_small`: 5% joint, 15% result, 5% tool, 24.71% micro.",
    "- `V8_2` is still better than the stable V8_0 baseline on the same 20 tasks, but the absolute score is far below a credible top1 candidate.",
    "- `V8_6_top1_candidate` all-helper mode regressed on smoke and should remain disabled.",
    "- `V8_1_order_helper` did not improve smoke versus V8_0, so it should not be expanded.",
    "",
    "## Failure Shape From Validation A Small",
    "",
    "- Order remains the largest process-trajectory gap: `V8_2` got only 1/13 order GT calls on the 5 held-out order1 tasks.",
    "- Retail generalization is weak on non-retail9 validation tasks: most retail A-small files have 0 result/tool success.",
    "- Restaurant validation tasks expose missing process coverage despite restaurant4 smoke success.",
    "- Kitchen improved locally but still has branch-selection and final-state misses outside the smoke task.",
    "",
    "## Top1 Readiness",
    "",
    "- Current plan is not top1-ready based on measurable dev holdout.",
    "- The final set has no GT, so final can only be sanity-checked for format/completeness, not self-scored.",
    "- A credible top1 attempt needs at minimum: stable validation_A and validation_B holdout gains, no retail/restaurant regression, and a final-compliant output package with complete histories.",
    "",
    "## Recommended Next Moves",
    "",
    "1. Mine validation_A failures into scenario-specific process templates without hardcoding task answers.",
    "2. Build per-scenario retrieval/canonicalization diagnostics for order/retail/restaurant before adding more LLM prior.",
    "3. Use visual/contact-sheet retry only for order/kitchen uncertainty cases, then measure on the same frozen split.",
    "4. Create SFT/teacher data from dev GT and successful traces for process-shape learning; do not use final metadata.",
    "5. Only run validation_B_holdout after validation_A_small exceeds the protected 4-task best on both joint and micro without retail/restaurant regressions.",
    "",
    "## Artifacts",
    "",
    "- Corrected V8 smoke reports: `reports/V8_SMOKE_SUMMARY_v8_smoke_20260617_1823_corrected.md`, `reports/V8_SMOKE_SUMMARY_v8_0_smoke_20260617_1832_corrected.md`.",
    "- New smoke reports: `reports/V8_SMOKE_SUMMARY_v8_1_smoke_20260617_continue.md`, `reports/V8_SMOKE_SUMMARY_v8_2_smoke_20260617_continue.md`.",
    "- Long validation reports: `reports/V8_VALIDATION_A_SMALL_v8_2_valA_small_20260617_continue.md`, `reports/V8_VALIDATION_A_SMALL_v8_0_valA_small_20260617_continue.md`.",
]
report.write_text("\n".join(lines) + "\n", encoding="utf-8")

state = {
    "updated_at": ts,
    "decision": "do_not_update_best",
    "protected_best": best,
    "latest_v8_readiness_report": str(report),
    "v8_rows": [{k: v for k, v in r.items() if k != "rows"} for r in rows],
    "next_recommended_version": "mine_validation_A_failures_before_more_expansion",
    "final_submission": "not_submitted",
}
(CODEX / "state" / "v8_top1_readiness_latest.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

with (CODEX / "README_STATUS.md").open("a", encoding="utf-8") as f:
    f.write(
        f"\n## V8 Top1 Readiness {ts}\n\n"
        f"- Report: `{report}`\n"
        f"- Decision: do not update protected best; no final submission.\n"
        f"- Best remains: `{best.get('version','')}` / `{best.get('run_id','')}`.\n"
        "- Fixed V8 validation micro extraction and recomputed smoke metrics.\n"
        "- V8_2 smoke improved to joint 0.75 / micro 0.8182, but validation_A_small was only joint 0.05 / result 0.15 / tool 0.05 / micro 0.2471.\n"
        "- Next: mine validation_A failures and improve order/retail/restaurant process coverage before validation_B or final.\n"
    )

print(report)
