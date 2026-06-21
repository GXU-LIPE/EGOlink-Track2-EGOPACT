#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


CODEX = Path("/home/data-gxu/acm/egolink2026-main/code/track2/codex")
EGO = Path("/home/data-gxu/acm/egolink2026-main/code/track2/EgoBench")
SPLIT = CODEX / "state" / "materialized_splits" / "validation_A_limit30"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--model-dir", required=True)
    args = ap.parse_args()
    sys.path.insert(0, str(EGO / "analysis_scripts"))
    sys.path.insert(0, str(EGO))
    from evaluate_interaction import evaluate_interaction_success
    import argparse as _argparse

    manifest = json.loads((SPLIT / "manifest.json").read_text())
    rows = []
    for scenario, num, _idxs in manifest.get("specs", []):
        spec = f"{scenario}{num}"
        result = Path(args.model_dir) / f"{spec}_easy.json"
        if not result.exists():
            continue
        metrics = evaluate_interaction_success(
            str(SPLIT / f"{spec}.json"),
            str(result),
            scenario=scenario,
            args=_argparse.Namespace(scenario_number=int(num)),
            silent=True,
            num_samples=0,
        )
        micro = metrics.get("micro_tool_stats", {}) or {}
        rows.append({
            "spec": spec,
            "scenario": scenario,
            "valid": metrics.get("valid_scenarios", 0),
            "joint": metrics.get("joint_success", {}).get("success_rate", 0),
            "result": metrics.get("result_based", {}).get("success_rate", 0),
            "tool": metrics.get("tool_based", {}).get("success_rate", 0),
            "micro": micro.get("micro_accuracy", 0),
            "correct_calls": micro.get("total_correct_calls", 0),
            "gt_calls": micro.get("total_ground_truth_calls", 0),
            "interaction_calls": micro.get("total_interaction_calls", 0),
        })
    valid = sum(r["valid"] for r in rows)
    correct = sum(r["correct_calls"] for r in rows)
    gt = sum(r["gt_calls"] for r in rows)
    def wavg(k):
        return sum(r[k] * r["valid"] for r in rows) / valid if valid else 0.0
    out = {
        "run_id": args.run_id,
        "partial": True,
        "evaluated_specs": len(rows),
        "valid": valid,
        "joint": wavg("joint"),
        "result": wavg("result"),
        "tool": wavg("tool"),
        "micro": correct / gt if gt else wavg("micro"),
        "correct_calls": correct,
        "gt_calls": gt,
        "interaction_calls": sum(r["interaction_calls"] for r in rows),
        "rows": rows,
    }
    path = CODEX / "runs" / "V16_gt100_distilled_policy_val41" / args.run_id / "partial_eval_summary.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
