#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Finalize Track2 V7 human-prior reports and README status."""

from __future__ import annotations

import json
import time
from pathlib import Path


CODEX = Path("/home/data-gxu/acm/egolink2026-main/code/track2/codex")


def read(path: Path, default: str = "") -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return default


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def append_once(path: Path, marker: str, text: str) -> None:
    old = read(path)
    if marker in old:
        return
    path.write_text(old.rstrip() + "\n\n" + text.strip() + "\n", encoding="utf-8")


def main() -> int:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    best = load_json(CODEX / "state" / "best_track2_api_version.json") or {}
    v71 = read(CODEX / "reports" / "02_gpt55_gate_summary_human_prior_v71c_20260617_1506.md")
    order_diag = read(CODEX / "reports" / "02_gpt55_gate_summary_human_prior_order1_diag_20260617_1548.md")
    order_log = read(CODEX / "runs" / "V7_1_process_verifier" / "human_prior_order1_diag_20260617_1548" / "logs" / "order1.log")
    policy_count = 0
    try:
        policy_count = sum(1 for _ in (CODEX / "train_data" / "human_prior_policy_traces.jsonl").open("r", encoding="utf-8"))
    except Exception:
        pass

    order_findings = [
        "order1 diagnostic: the agent first asked the simulated user for visual dish/category names when contact_sheet was missing.",
        "After user pushback it used retrieval tools, added Turkey breast ham, and reached the aggregate stage.",
        "It then repeated compute_total_payment after a 0.0 return; db_guard now records all final compute attempts, including unsuccessful 0.0 results, to block same-parameter loops.",
        "Human-prior prompt now says order tasks with missing contact sheet must not ask the simulated user for visual names; use image_description/task analysis/layout hint plus pinned-restaurant retrieval.",
    ]

    impl = CODEX / "reports" / f"HUMAN_PRIOR_IMPLEMENTATION_{stamp}.md"
    impl.write_text("\n".join([
        "# Human Prior Implementation Addendum",
        "",
        f"- generated_at: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}",
        "- base_version: V6_1_3_gpt55_guarded_endpoint",
        "- implemented_version_tested: V7_1_process_verifier",
        "- final_auto_submitted: no",
        "- api_key_logged: no",
        "",
        "## Implemented Modules",
        "",
        "- `tool_affordance_memory.py`: schema-derived tool families and stage candidate caps.",
        "- `human_process_graph.py`: retail/kitchen/restaurant/order process-stage prior.",
        "- `process_coverage_verifier.py`: shape-level process coverage checks without GT answers.",
        "- `counterfactual_db_simulator.py`: dry-run risk checks for state-changing calls.",
        "- `visual_slot_prior.py`: cached visual_state/contact-sheet slots as candidates only.",
        "- `working_memory_manager.py`: compact prompt memory with pins, stage, ledgers, recent turns, and slots.",
        "- `human_prior_controller.py`: V7 event logging and policy trace generation.",
        "",
        "## Targeted Fixes During This Run",
        "",
        "- Added balanced JSON-prefix repair for mixed `JSON + text` outputs.",
        "- Added per-scenario timeout and unbuffered Python logging for V7 gates.",
        "- Narrowed retail aggregate intent so `lowest calories` as a filter does not force final aggregate.",
        "- Added order no-visual/no-follow-up guidance and compute-loop ledger recording.",
        "",
        f"- policy_trace_count: {policy_count}",
    ]) + "\n", encoding="utf-8")

    gate = CODEX / "reports" / f"HUMAN_PRIOR_GATE_SUMMARY_{stamp}.md"
    gate.write_text("\n".join([
        "# Human Prior Gate Summary Addendum",
        "",
        f"- generated_at: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}",
        "- run_id: human_prior_v71c_20260617_1506",
        "- version: V7_1_process_verifier",
        "- final_auto_submitted: no",
        "- api_key_logged: no",
        "- best_state_updated: no",
        "",
        "## Result",
        "",
        "- V7_1 4-task gate: joint 50%, result 50%, tool 50%, micro 62.5%, avg tool calls 19.5.",
        "- V6_1_3 remains best: joint 50%, result 75%, tool 50%, micro 70.83%.",
        "- V7_1 preserved retail9 and restaurant4 joint success.",
        "- order1 failed in the 4-task gate due timeout/empty output, then order1-only diagnostic showed progress but still no success.",
        "- kitchen2 remained failed: micro 50%, result 0%, 46 tool calls.",
        "",
        "## V7_1 Gate Source",
        "",
        v71.strip(),
        "",
        "## order1 Diagnostic",
        "",
        order_diag.strip(),
        "",
        "## Key Diagnosis",
        "",
        "\n".join(f"- {x}" for x in order_findings),
    ]) + "\n", encoding="utf-8")

    ablation = CODEX / "reports" / f"HUMAN_PRIOR_ABLATION_{stamp}.md"
    ablation.write_text("\n".join([
        "# Human Prior Ablation Addendum",
        "",
        f"- generated_at: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}",
        "- baseline: V6_1_3_gpt55_guarded_endpoint",
        "- tested: V7_1_process_verifier",
        "- final_auto_submitted: no",
        "",
        "## Ablation Interpretation",
        "",
        "- Human-prior telemetry and policy traces are functional.",
        "- JSON repair improved structural robustness for mixed tool/text outputs.",
        "- Stage prompts reduced some drift but can over-constrain kitchen and cause order visual follow-up unless guarded.",
        "- Counterfactual/pre-execution blocks were kept off for V7_1; V7_2+ should use them selectively after order/kitchen helper refinements.",
        "- V7_1 did not meet best-update criteria because result success and micro were below V6_1_3.",
        "",
        "## Recommended Next V7 Slice",
        "",
        "- V7_3_order_kitchen_helpers should be next, not V7_4 full.",
        "- order: inject task analysis/layout hint into human-prior working memory; forbid visual-detail follow-up when no contact sheet; detect compute_total_payment 0.0 loops.",
        "- kitchen: delay conservative mode for filtered stock-quantity queries; allow recipe/menu/fridge intersection quantities even after >35 calls when they are pending branch-critical queries.",
    ]) + "\n", encoding="utf-8")

    paper = CODEX / "reports" / f"HUMAN_PRIOR_PAPER_NOTES_{stamp}.md"
    paper.write_text("\n".join([
        "# Human-Prior Tool Agent for Egocentric Service Tasks",
        "",
        f"- generated_at: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}",
        "- Track2 allows commercial API; GPT-5.5 is used as the service agent through the OpenAI-compatible ai-pixel endpoint.",
        "- final_auto_submitted: no",
        "- claim boundary: motivated by human cognitive priors; no claim of medical/neuroscience causal proof.",
        "",
        "## Method Components",
        "",
        "- Human Process Graph: scenario stage templates for process coverage.",
        "- Tool Affordance Memory: schema-derived read/mutate/aggregate tags and timing hints.",
        "- Visual-to-Slot Prior: cached visual evidence converted to candidate slots, verified via tools.",
        "- Counterfactual DB Simulator: pre-execution sanity checks from pins and mutation ledger.",
        "- Process-Coverage Verifier: detects missing aggregate/mutation stages without GT answers.",
        "- Working Memory Manager: caps active tool/entity candidates and summarizes ledgers/history.",
        "- Socially Robust User Guidance: short replies, fewer follow-up questions, no asking for unavailable visual details.",
        "",
        "## Empirical Notes",
        "",
        "- V7_1 preserved successful retail9 and restaurant4 trajectories.",
        "- V7_1 did not improve aggregate 4-task score over V6_1_3.",
        "- order1 diagnostic suggests the method is useful for process-stage tracing, but needs layout-hint grounding.",
        "- kitchen2 diagnostic suggests conservative-mode thresholds need branch-aware exceptions.",
    ]) + "\n", encoding="utf-8")

    readme_marker = "## 2026-06-17 V7 Human-Prior Agent"
    readme_text = f"""
{readme_marker}

- V7 modules installed under `wrappers/egobench_agent_plus/`; official EgoBench source was not modified.
- Tested run: `V7_1_process_verifier` / `human_prior_v71c_20260617_1506`.
- V7_1 4-task metrics: joint 50%, result 50%, tool 50%, micro 62.5%, avg tool calls 19.5.
- Current best remains `{best.get('version')}` / `{best.get('run_id')}` with joint {best.get('joint_success')}, result {best.get('result_success')}, tool {best.get('tool_success')}, micro {best.get('micro_tool_accuracy')}.
- V7 preserved retail9 and restaurant4 joint success, but did not improve order1/kitchen2 enough to replace best.
- New traces: `train_data/human_prior_policy_traces.jsonl` ({policy_count} records).
- Reports: `{impl.name}`, `{gate.name}`, `{ablation.name}`, `{paper.name}`.
- Final submission: not submitted.
"""
    append_once(CODEX / "README_STATUS.md", readme_marker, readme_text)

    state = {
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "tested_version": "V7_1_process_verifier",
        "tested_run_id": "human_prior_v71c_20260617_1506",
        "best_updated": False,
        "best_version": best.get("version"),
        "reports": [str(impl), str(gate), str(ablation), str(paper)],
        "policy_trace_count": policy_count,
    }
    (CODEX / "state" / "latest_human_prior_status.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(state, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
