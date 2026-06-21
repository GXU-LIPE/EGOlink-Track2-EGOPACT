#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Finalize V20 reports after single-sample and clean-retail runs."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

CODEX = Path("/home/data-gxu/acm/egolink2026-main/code/track2/codex")
RUN_SINGLE = "v20_single_retail_chain_20260620_1708"
RUN_THREE = "v20_clean_retail_three_20260620_1720"


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return default


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    single = read_json(CODEX / "state" / "latest_v20_single_retail_chain.json", {})
    three = read_json(CODEX / "state" / "latest_v20_clean_retail_three.json", {})
    trace = read_json(CODEX / "analysis" / "v20_single_retail_chain_trace.json", {})
    three_summary = three.get("summary", {})
    report = CODEX / "reports" / f"V20_NEXT_DECISION_{RUN_THREE}.md"

    def fmt_summary(label: str) -> str:
        row = three_summary.get(label, {})
        return f"{row.get('joint', 0):.2%} joint, {row.get('micro', 0):.4f} micro, {row.get('matched_tools', 0)}/{row.get('gt_tools', 0)} tools"

    v19_break = ((trace.get("gt_diff") or {}).get("B_v19_original_case_reuse") or {}).get("first_failure_point", {})
    report.write_text(
        "\n".join(
            [
                f"# V20 Next Decision {RUN_THREE}",
                "",
                "## Single-Sample Surgery",
                "",
                "- selected sample: `retail4::14`",
                "- reason: clean retail4 sample, instruction/analysis/GT/video consistent, GT replay joint100, V14/V19 non-joint with non-empty trajectory.",
                f"- V19 first break: `{v19_break.get('kind')}` at index `{v19_break.get('idx')}`.",
                "- V19 failure locus: selected fallback trajectory starts with broad/retrieval tools for the wrong branch (`find_products_by_country_of_origin`) instead of current-task visual/entity process (`get_taste`, `get_country_of_origin`).",
                "- V20 repair target: current utterance user_id, visual/product canonicalization, no foreign entity copy, no broad scan, aggregate closure.",
                "",
                "Single-sample metrics:",
                "",
                "- V19: 0/4 matched tools, joint false.",
                "- V20 non-oracle resolver: 2/4 matched tools, micro 0.5000, joint false.",
                "- V20 repair diagnostic: 4/4 matched tools, joint true.",
                "",
                "## Clean Retail Tiny Expansion",
                "",
                "Scope: clean `retail4`, `retail2`, `retail1` only. This is not full val41.",
                "",
                "| candidate | summary |",
                "|---|---:|",
                f"| V14 | {fmt_summary('V14')} |",
                f"| V19 | {fmt_summary('V19')} |",
                f"| V20_nonoracle | {fmt_summary('V20_nonoracle')} |",
                f"| V20_repair | {fmt_summary('V20_repair')} |",
                "",
                "## Interpretation",
                "",
                "V20 proves the case-reuse path can be surgically improved. The non-oracle resolver already improves micro by restoring the front of the process and shortening the trajectory, but it still cannot infer branch target/add-product slots reliably. The repair diagnostic shows the compiler/evaluator path can reach high joint if branch-action slot filling is supplied.",
                "",
                "Next useful work: replace the diagnostic repair hint with a non-oracle branch target resolver using tool observations and constrained DB queries. Do not return to broad GT100 distillation until that branch slot resolver works on these clean retail samples.",
                "",
                "## Boundary",
                "",
                "- full_val41_run: false",
                "- final_run: false",
                "- v10_zip_overwritten: false",
                "- auto_submit: false",
                "- uses_final_hidden_metadata: false",
                "",
                "## Artifacts",
                "",
                f"- single trace: `{CODEX / 'analysis' / 'v20_single_retail_chain_trace.json'}`",
                f"- single candidates: `{CODEX / 'analysis' / 'v20_single_retail_candidates.jsonl'}`",
                f"- clean retail trace: `{CODEX / 'analysis' / 'v20_clean_retail_three_trace.json'}`",
                f"- clean retail report: `{CODEX / 'reports' / ('V20_CLEAN_RETAIL_THREE_EVAL_' + RUN_THREE + '.md')}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    state = {
        "run_id": RUN_THREE,
        "single_run_id": RUN_SINGLE,
        "single_state": single,
        "clean_retail_state": {
            "run_id": three.get("run_id"),
            "summary": three_summary,
            "full_val41_run": three.get("full_val41_run"),
            "final_run": three.get("final_run"),
            "v10_zip_overwritten": three.get("v10_zip_overwritten"),
        },
        "decision_report": str(report),
        "recommendation": "build non-oracle branch target resolver before any larger run",
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    write_json(CODEX / "state" / "latest_v20_summary.json", state)
    print(json.dumps({"ok": True, "report": str(report), "state": str(CODEX / "state" / "latest_v20_summary.json")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
