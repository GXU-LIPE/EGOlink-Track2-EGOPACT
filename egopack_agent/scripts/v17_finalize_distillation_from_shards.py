#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


CODEX = Path("/home/data-gxu/acm/egolink2026-main/code/track2/codex")
GT17 = CODEX / "gt_distill_v17"


def read_jsonl(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    tmp.replace(path)


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_report(run_id: str, summary):
    report = CODEX / "reports" / f"V17_GPT55_DISTILLATION_CHECK_{run_id}.md"
    lines = [
        f"# V17 GPT-5.5 Distillation Check {run_id}",
        "",
        f"- status: {summary['status']}",
        "- final_run: false",
        "- uses_final_hidden_metadata: false",
        "- uses_val41_label: false",
        "- deterministic_fallback_allowed: false",
        "- source: recovered from completed shard jsonl files",
        "",
        "## Metrics",
        "",
        f"- pool_rows: {summary['pool_rows']}",
        f"- attempted_rows: {summary['attempted_rows']}",
        f"- gpt55_rows: {summary['gpt55_rows']}",
        f"- invalid_rows: {summary['invalid_rows']}",
        f"- parse_valid_rate: {summary['parse_valid_rate']:.4f}",
        f"- scenario_coverage: {summary['scenario_coverage']}",
        f"- shard_jsonl_count: {summary['shard_jsonl_count']}",
    ]
    if summary.get("fail_reasons"):
        lines += ["", "## Fail Reasons", ""]
        lines += [f"- {x}" for x in summary["fail_reasons"]]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()
    shard_dir = GT17 / "gpt55_shards" / args.run_id
    rows = []
    for p in sorted(shard_dir.glob("shard_*.jsonl")):
        rows.extend(read_jsonl(p))
    valid = [r for r in rows if r.get("valid") is True]
    scenarios = sorted({r.get("scenario") for r in valid if r.get("scenario")})
    rate = len(valid) / len(rows) if rows else 0.0
    fail = []
    if len(valid) < 600:
        fail.append(f"gpt55_rows {len(valid)} < 600")
    if rate < 0.95:
        fail.append(f"parse_valid_rate {rate:.4f} < 0.95")
    missing = sorted(set(["order", "restaurant", "retail", "kitchen"]) - set(scenarios))
    if missing:
        fail.append("missing scenario coverage: " + ",".join(missing))
    out = GT17 / "gpt55_distilled_rules.jsonl"
    write_jsonl(out, rows)
    summary = {
        "run_id": args.run_id,
        "status": "PASS" if not fail else "FAIL",
        "pool_rows": 696,
        "attempted_rows": len(rows),
        "gpt55_rows": len(valid),
        "invalid_rows": len(rows) - len(valid),
        "parse_valid_rate": rate,
        "scenario_coverage": scenarios,
        "shard_jsonl_count": len(list(shard_dir.glob("shard_*.jsonl"))),
        "fail_reasons": fail,
        "output": str(out),
        "uses_final_hidden_metadata": False,
        "uses_val41_label": False,
        "deterministic_fallback_allowed": False,
        "recovered_from_shards": True,
        "finalized_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    report = write_report(args.run_id, summary)
    summary["report"] = str(report)
    write_json(GT17 / "gpt55_distillation_manifest.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if fail:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
