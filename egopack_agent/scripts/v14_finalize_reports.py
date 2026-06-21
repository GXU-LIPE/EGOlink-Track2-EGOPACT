#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Finalize V14 reports."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path


CODEX = Path(os.environ.get("CODEX_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/codex"))


GT_RUN = "v14_gt_distill_20260619_211502"
DIST_RUN = "v14_distilled_val41_20260619_211502"
SEL_RUN = "v14_candidate_selection_20260619_2134"


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def pct(x: float) -> str:
    return f"{100*x:.2f}%"


def main() -> None:
    gt = read_json(CODEX / "state" / "latest_v14_gt_distillation.json")
    dist = read_json(CODEX / "state" / "latest_v14_distilled_val41.json")
    sel = read_json(CODEX / "state" / "latest_v14_candidate_selection_val41.json")
    dist_eval = read_json(CODEX / "runs" / "V14_val41_distilled_no_task_oracle" / DIST_RUN / "eval_summary.json")
    sel_eval = read_json(CODEX / "runs" / "V14_candidate_selection_val41" / SEL_RUN / "eval_summary.json")

    dist_order = next(r for r in dist_eval["rows"] if r["spec"] == "order1")
    sel_order = next(r for r in sel_eval["rows"] if r["spec"] == "order1")

    order_report = CODEX / "reports" / f"V14_ORDER1_REPAIR_{DIST_RUN}.md"
    write(
        order_report,
        "\n".join(
            [
                f"# V14 Order1 Repair {DIST_RUN}",
                "",
                "- final_run: false",
                "- uses_val41_gt_directly: false for B tier; A tier oracle uses GT only for upper-bound debugging",
                "",
                "## Distilled B Tier",
                "",
                f"- order1 valid: {dist_order.get('valid')}",
                f"- order1 joint: {pct(dist_order.get('joint', 0))}",
                f"- order1 result: {pct(dist_order.get('result', 0))}",
                f"- order1 tool: {pct(dist_order.get('tool', 0))}",
                f"- order1 micro: {pct(dist_order.get('micro', 0))}",
                f"- order1 tool matches: {dist_order.get('correct_calls')}/{dist_order.get('gt_calls')}",
                "",
                "## Candidate Selection",
                "",
                f"- order1 joint: {pct(sel_order.get('joint', 0))}",
                f"- order1 result: {pct(sel_order.get('result', 0))}",
                f"- order1 tool: {pct(sel_order.get('tool', 0))}",
                f"- order1 micro: {pct(sel_order.get('micro', 0))}",
                f"- order1 tool matches: {sel_order.get('correct_calls')}/{sel_order.get('gt_calls')}",
                "",
                "## Diagnosis",
                "",
                "- V14 process policy improved order1 micro from V12 3/24 to 4/24, but joint stayed 0/8.",
                "- Remaining blocker is entity/visual slot selection and branch execution, not evaluator/replay format.",
                "- The oracle replay gets order1 8/8 joint, proving the official evaluator accepts the GT-shaped executable trajectory.",
                "",
            ]
        ),
    )

    final_report = CODEX / "reports" / f"V14_FINAL_SUMMARY_{time.strftime('%Y%m%d_%H%M%S')}.md"
    lines = [
        "# V14 GT Trajectory Distillation Val41 Summary",
        "",
        "- final_run: false",
        "- V10 protected zip overwritten: false",
        "- automatic final submission: false",
        "- API key logged: false",
        "",
        "## A Tier Oracle Teacher",
        "",
        "- Uses val41 GT: yes, explicit upper-bound/debug only.",
        f"- joint: {pct(gt['oracle_summary']['joint'])}",
        f"- result: {pct(gt['oracle_summary']['result'])}",
        f"- tool: {pct(gt['oracle_summary']['tool'])}",
        f"- micro: {pct(gt['oracle_summary']['micro'])}",
        f"- report: `{gt['reports']['oracle']}`",
        "",
        "## B Tier Distilled No Task Oracle",
        "",
        "- Uses val41 GT directly: no.",
        "- Uses distilled process bank: yes.",
        f"- joint: {pct(dist['summary']['joint'])}",
        f"- result: {pct(dist['summary']['result'])}",
        f"- tool: {pct(dist['summary']['tool'])}",
        f"- micro: {pct(dist['summary']['micro'])}",
        f"- interaction_calls: {dist['summary']['interaction_calls']}",
        f"- report: `{dist['report']}`",
        "",
        "## Val41 Candidate Selection",
        "",
        "- Uses val41 GT for evaluator selection: yes, debug-only.",
        "- Oracle candidate included: no.",
        f"- chosen_counts: `{json.dumps(sel.get('chosen_counts', {}), ensure_ascii=False)}`",
        f"- joint: {pct(sel['summary']['joint'])}",
        f"- result: {pct(sel['summary']['result'])}",
        f"- tool: {pct(sel['summary']['tool'])}",
        f"- micro: {pct(sel['summary']['micro'])}",
        f"- interaction_calls: {sel['summary']['interaction_calls']}",
        f"- report: `{sel['report']}`",
        "",
        "## Mismatch Audit",
        "",
        "- branch condition wrong: 41/41",
        "- broad scan instead of constrained candidate: 31/41",
        "- tool type wrong: 20/41",
        "- overlong trajectory / loop: 19/41",
        "- visual entity wrong / canonical mismatch: 14/41",
        "- missing final aggregate: 6/41",
        "- restaurant/user pin wrong: 5/41",
        f"- audit report: `{gt['reports']['audit']}`",
        "",
        "## Answer to Goals",
        "",
        "- A tier oracle upper bound reached 100% joint, so evaluator/replay format is correct.",
        "- B tier distilled improved over V12: joint 12.20% -> 14.63%, micro 29.49% -> 33.97%. It did not reach 19.43%.",
        "- Candidate selection improved micro to 36.54% and lowered interaction calls to 795, but joint stayed 14.63%.",
        "- order1 improved only slightly: 3/24 -> 4/24 micro, still 0 joint.",
        "- Tasks converted to additional joint success in B tier: kitchen3 gained 1/4 joint compared with V12; other joint wins remained mostly kitchen2/restaurant4/retail3/retail4.",
        "- Not ready for final-style online version until order1/restaurant3/retail7/retail8 entity slot filling is fixed.",
        "",
    ]
    write(final_report, "\n".join(lines))
    state = {
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "gt_run": GT_RUN,
        "distilled_run": DIST_RUN,
        "candidate_selection_run": SEL_RUN,
        "reports": {
            "final_summary": str(final_report),
            "order1": str(order_report),
            "audit": gt["reports"]["audit"],
            "oracle": gt["reports"]["oracle"],
            "distilled": dist["report"],
            "candidate_selection": sel["report"],
        },
        "recommendation": "Do not run final. Continue with executable slot-filling repair for order1/restaurant3/retail7/retail8.",
        "final_run": False,
        "v10_zip_overwritten": False,
    }
    (CODEX / "state" / "latest_v14_summary.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
