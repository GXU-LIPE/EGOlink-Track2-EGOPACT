#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import time
from pathlib import Path


CODEX = Path("/home/data-gxu/acm/egolink2026-main/code/track2/codex")


def load(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return default


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_id = "v16_gt100_val41_20260619_2352"
    raw = load(CODEX / "runs" / "V16_gt100_distilled_policy_val41" / run_id / "partial_eval_summary.json", {})
    manifest = load(CODEX / "gt_distill_v16" / "gt100_pool_manifest.json", {})
    select = load(CODEX / "state" / "latest_v16_candidate_selection.json", {})
    distill = load(CODEX / "gt_distill_v16" / "gpt55_rule_distillation_summary.json", {})

    val_report = CODEX / "reports" / f"V16_VAL41_RESULT_{run_id}.md"
    lines = [
        f"# V16 Val41 Result {run_id}",
        "",
        "- version: V16_GT100_TRAJECTORY_DISTILL_ALL_NONFINAL_NONVAL41",
        "- final_run: false",
        "- uses_final_hidden_metadata: false",
        "- uses_val41_gt_for_policy: false",
        "- uses_val41_gt_for_candidate_selection: false",
        "- protected_v10_zip_overwritten: false",
        "",
        "## GT100 Pool",
        "",
        f"- total_pool_rows: {manifest.get('total_pool_rows')}",
        f"- scenario_joint100_kept: {manifest.get('scenario_joint100_kept')}",
        f"- train_data_fallback_kept: {manifest.get('train_data_fallback_kept')}",
        f"- excluded_rows: {manifest.get('excluded_rows')}",
        f"- excludes_official_final309: {manifest.get('excludes_official_final309')}",
        f"- excludes_frozen_val41: {manifest.get('excludes_frozen_val41')}",
        "",
        "## GPT-5.5 Rule Distillation",
        "",
        f"- samples_completed: {distill.get('samples_completed', 0)}",
        f"- gpt55_rows: {distill.get('gpt55_rows', 0)}",
        f"- fallback_rows: {distill.get('fallback_rows', 0)}",
        f"- status: {distill.get('reason', 'completed')}",
        "",
        "## V16 Raw Val41",
        "",
        f"- valid: {raw.get('valid')}",
        f"- joint: {raw.get('joint', 0):.4f}",
        f"- result: {raw.get('result', 0):.4f}",
        f"- tool: {raw.get('tool', 0):.4f}",
        f"- micro: {raw.get('micro', 0):.4f}",
        f"- tool_call_match_counts: {raw.get('correct_calls')}/{raw.get('gt_calls')} gt, interaction_calls={raw.get('interaction_calls')}",
        "",
        "## V16 Non-Oracle Candidate Selection",
        "",
    ]
    ss = select.get("summary") or {}
    lines += [
        f"- joint: {ss.get('joint', 0):.4f}",
        f"- result: {ss.get('result', 0):.4f}",
        f"- tool: {ss.get('tool', 0):.4f}",
        f"- micro: {ss.get('micro', 0):.4f}",
        f"- tool_call_match_counts: {ss.get('correct_calls')}/{ss.get('gt_calls')} gt, interaction_calls={ss.get('interaction_calls')}",
        f"- result_dir: `{select.get('result_dir')}`",
        "",
        "## Per Spec Raw V16",
        "",
        "| spec | valid | joint | result | tool | micro | calls | interaction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in raw.get("rows", []):
        lines.append(f"| {row.get('spec')} | {row.get('valid')} | {row.get('joint',0):.3f} | {row.get('result',0):.3f} | {row.get('tool',0):.3f} | {row.get('micro',0):.3f} | {row.get('correct_calls')}/{row.get('gt_calls')} | {row.get('interaction_calls')} |")
    write(val_report, "\n".join(lines) + "\n")

    decision = CODEX / "reports" / f"V16_NEXT_DECISION_{stamp}.md"
    v14_joint = 0.1463
    v14_micro = 0.3654
    raw_joint = raw.get("joint", 0)
    raw_micro = raw.get("micro", 0)
    sel_joint = ss.get("joint", 0)
    sel_micro = ss.get("micro", 0)
    recommend = "do_not_promote_v16"
    if sel_joint > v14_joint and sel_micro > v14_micro:
        recommend = "consider_v16_next"
    decision_lines = [
        f"# V16 Next Decision {stamp}",
        "",
        "- final_run: false",
        "- no_auto_submit: true",
        "- protected_best_updated: false",
        "",
        "## Baselines",
        "",
        "- V14 distilled: joint 14.63%, micro 33.97%",
        "- V14 candidate selection: joint 14.63%, micro 36.54%",
        "",
        "## V16",
        "",
        f"- raw joint: {raw_joint:.4f}",
        f"- raw micro: {raw_micro:.4f}",
        f"- selected joint: {sel_joint:.4f}",
        f"- selected micro: {sel_micro:.4f}",
        f"- order1 raw micro: {next((r.get('micro') for r in raw.get('rows', []) if r.get('spec') == 'order1'), None)}",
        "",
        "## Decision",
        "",
        f"- recommendation: {recommend}",
        "- reason: V16 matched V14 joint but did not beat V14 candidate micro; order1 regressed to 3/24 matched calls.",
        "- next technical direction: use GT100 modules for targeted compiler/slot resolver code, not just prompt injection. Specifically implement executable order1 branch compiler and retail7/8 visual slot resolver before another full val41 run.",
        "",
        "## Required Questions Answered",
        "",
        f"1. GT100 pool rows: {manifest.get('total_pool_rows')}.",
        "2. final309 and val41 were excluded from distillation: yes.",
        "3. Executable modules distilled: tool_sequence_automata, slot_resolver_rules, scenario_entity_lexicon, anti_failure_rules, process_repair_templates, candidate_rerank_weights, V16 prompt policy modules.",
        f"4. Val41 exceeded V14: no; joint ties V14 at {raw_joint:.4f}, micro below V14 candidate.",
        "5. New joint successes: no confirmed net gain from V14 at aggregate level.",
        "6. order1 improved: no; raw order1 micro is 3/24, worse than V14's 4/24.",
        "7. Baseline 19.43% reached: no.",
        "8. Worth entering final-style online version: not yet; keep V10 protected, use V16 only as distillation asset.",
    ]
    write(decision, "\n".join(decision_lines) + "\n")
    state = {
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "version": "V16_GT100_TRAJECTORY_DISTILL_ALL_NONFINAL_NONVAL41",
        "raw_report": str(val_report),
        "decision_report": str(decision),
        "raw_summary": {k: raw.get(k) for k in ["valid", "joint", "result", "tool", "micro", "correct_calls", "gt_calls", "interaction_calls"]},
        "selected_summary": ss,
        "recommendation": recommend,
        "final_run": False,
        "protected_best_updated": False,
    }
    (CODEX / "state" / "latest_v16_summary.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
