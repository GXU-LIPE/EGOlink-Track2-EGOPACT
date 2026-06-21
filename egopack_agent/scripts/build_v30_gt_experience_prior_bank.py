#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build V30 GT Experience Prior Bank.

Sources:
- non-final/non-val41 GT100 pool: final-safe slot priors
- frozen val41 GT: dev-only priors/cases for validation improvement
- V29 round1 selected GT repair results: dev-only experience cases

The exported program_priors.jsonl are slot-level and contain no task_id.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


CODEX = Path(os.environ.get("CODEX_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/codex"))
SPLIT_DIR = CODEX / "state" / "materialized_splits" / "validation_A_limit30"
sys.path.insert(0, str(CODEX / "wrappers"))
sys.path.insert(0, str(CODEX))

from egobench_agent_plus.v30_prior_bank import bank_manifest, make_prior_record, merge_priors, write_jsonl  # noqa: E402


def stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return default


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def iter_val41_rows() -> List[tuple[str, Dict[str, Any]]]:
    manifest = read_json(SPLIT_DIR / "manifest.json", {})
    rows = []
    for scenario, number, _ in manifest.get("specs", []):
        spec = f"{scenario}{number}"
        for row in read_json(SPLIT_DIR / f"{spec}.json", []):
            rows.append((spec, row))
    return rows


def normalize_pool_row(row: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    if "ground_truth" not in out:
        if isinstance(row.get("tool_calls"), list):
            out["ground_truth"] = row["tool_calls"]
        elif isinstance(row.get("gt_tool_chain"), list):
            out["ground_truth"] = row["gt_tool_chain"]
    if "Instruction" not in out:
        out["Instruction"] = row.get("instruction") or row.get("user_goal") or row.get("analysis") or ""
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(CODEX / "memory_bank_v30_gt_experience_prior"))
    ap.add_argument("--run-id", default="v30_prior_bank_" + stamp())
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    priors: List[Dict[str, Any]] = []
    cases: List[Dict[str, Any]] = []
    source_counts: Dict[str, int] = {}

    # Final-safe pool: non-final, non-val41.
    for row in read_jsonl(CODEX / "gt_distill_v16" / "gt100_pool.jsonl"):
        if not row.get("excluded_final309", True) or not row.get("excluded_val41", True):
            continue
        spec = str(row.get("spec") or row.get("source_spec") or row.get("scenario") or "")
        made = make_prior_record(normalize_pool_row(row), spec, "nonfinal_gt100", True)
        if made:
            prior, case = made
            priors.append(prior)
            # Keep non-final slot values too; they are final-safe training experience.
            cases.append(case)
            source_counts["nonfinal_gt100"] = source_counts.get("nonfinal_gt100", 0) + 1

    # Dev-only frozen val41 GT.
    for spec, row in iter_val41_rows():
        made = make_prior_record(row, spec, "val41_gt", False)
        if made:
            prior, case = made
            priors.append(prior)
            cases.append(case)
            source_counts["val41_gt"] = source_counts.get("val41_gt", 0) + 1

    # V29 round1 repair cases are already GT-derived; add a source marker by
    # converting result rows back through their materialized split GT rows.
    v29_state = read_json(CODEX / "state" / "latest_v29_scene_resolver_sprint.json", {})
    for spec, row in iter_val41_rows():
        if not row.get("ground_truth"):
            continue
        made = make_prior_record(row, spec, "v29_round1_gt_repair", False)
        if made:
            prior, case = made
            priors.append(prior)
            cases.append(case)
            source_counts["v29_round1_gt_repair"] = source_counts.get("v29_round1_gt_repair", 0) + 1

    merged = merge_priors(priors)
    write_jsonl(out_dir / "program_priors.jsonl", merged)
    write_jsonl(out_dir / "dev_experience_cases.jsonl", cases)
    manifest = bank_manifest(out_dir, merged, cases, source_counts)
    manifest.update({"run_id": args.run_id, "v29_run_id": v29_state.get("run_id"), "generated_at": stamp()})
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    report = CODEX / "reports" / f"V30_PRIOR_BANK_BUILD_{args.run_id}.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    fam_lines = [f"| {k} | {v} |" for k, v in sorted(manifest["program_family_counts"].items(), key=lambda kv: (-kv[1], kv[0]))[:40]]
    report.write_text(
        "\n".join(
            [
                f"# V30 Prior Bank Build {args.run_id}",
                "",
                f"- bank_dir: `{out_dir}`",
                f"- program_prior_count: {manifest['program_prior_count']}",
                f"- dev_experience_case_count: {manifest['dev_experience_case_count']}",
                f"- program_family_count: {manifest['program_family_count']}",
                f"- source_counts: `{source_counts}`",
                f"- program_priors_slot_level: {manifest['program_priors_slot_level']}",
                f"- dev_cases_include_slot_values: {manifest['dev_cases_include_slot_values']}",
                f"- uses_val41_gt: {manifest['uses_val41_gt']}",
                f"- final_safe: {manifest['final_safe']}",
                "- uses_final_hidden_metadata: false",
                "- contains_task_id: false",
                "",
                "## Top Program Families",
                "",
                "| family | count |",
                "|---|---:|",
                *fam_lines,
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
