#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

CODEX = Path("/home/data-gxu/acm/egolink2026-main/code/track2/codex")


def load_manifest(card_dir: Path) -> dict[str, Any]:
    manifests = sorted(card_dir.glob("manifest_all_dev_offline_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not manifests:
        return {}
    return json.loads(manifests[0].read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--card_dir", required=True)
    parser.add_argument("--report_prefix", default="V12_ALL_DEV_OFFLINE_QWEN3VL_GROUNDING_AUDIT")
    args = parser.parse_args()

    card_dir = Path(args.card_dir)
    manifest = load_manifest(card_dir)
    expected_items = manifest.get("items") or []
    expected_names = {Path(item["path"]).name for item in expected_items if item.get("path")}
    card_paths = sorted(p for p in card_dir.glob("*.json") if not p.name.startswith("manifest") and "summary" not in p.name)
    path_by_name = {p.name: p for p in card_paths}
    names = set(path_by_name)
    missing = sorted(expected_names - names)
    extra = sorted(names - expected_names)

    failures: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    teacher_counts: Counter[str] = Counter()
    parse_counts: Counter[str] = Counter()
    scenario_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    ok_count = 0
    top_k_total = 0
    gpt55_cleaned_count = 0
    for name in sorted(expected_names & names):
        path = path_by_name[name]
        try:
            card = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            card = {"status": "grounding_failed", "teacher": "qwen3vl", "top_k_candidates": [], "error": f"parse:{type(exc).__name__}"}
        topk = card.get("top_k_candidates") or []
        status = str(card.get("status") or "")
        teacher = str(card.get("teacher") or "")
        parse_status = str(card.get("_qwen3vl_json_parse_status") or "")
        scenario = str(card.get("scenario") or "unknown")
        source = str(card.get("source_split") or "unknown")
        status_counts[status] += 1
        teacher_counts[teacher] += 1
        parse_counts[parse_status] += 1
        scenario_counts[scenario] += 1
        source_counts[source] += 1
        top_k_total += len(topk)
        if card.get("_gpt55_cleaned"):
            gpt55_cleaned_count += 1
        if status == "qwen3vl_success" and teacher == "qwen3vl" and topk:
            ok_count += 1
        else:
            failures.append(
                {
                    "cache_key": card.get("cache_key") or path.stem,
                    "status": status,
                    "teacher": teacher,
                    "parse_status": parse_status,
                    "top_k_count": len(topk),
                    "scenario": scenario,
                    "task_id": card.get("task_id"),
                    "uncertainty_notes": card.get("uncertainty_notes", [])[:5] if isinstance(card.get("uncertainty_notes"), list) else card.get("uncertainty_notes"),
                    "path": str(path),
                }
            )

    ts = time.strftime("%Y%m%d_%H%M%S")
    report_json = CODEX / "reports" / f"{args.report_prefix}_{ts}.json"
    report_md = CODEX / "reports" / f"{args.report_prefix}_{ts}.md"
    result = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "card_dir": str(card_dir),
        "expected_count": len(expected_items),
        "actual_expected_count": len(expected_names & names),
        "ok_count": ok_count,
        "missing_count": len(missing),
        "failed_count": len(failures),
        "extra_count": len(extra),
        "top_k_total": top_k_total,
        "gpt55_cleaned_count": gpt55_cleaned_count,
        "status_counts": dict(status_counts),
        "teacher_counts": dict(teacher_counts),
        "parse_status_counts": dict(parse_counts),
        "scenario_counts": dict(scenario_counts),
        "source_counts": dict(source_counts),
        "excluded_final_submission_specs": manifest.get("excluded_final_submission_specs", []),
        "excluded_files": manifest.get("excluded_files", []),
        "final_hidden_metadata_used": False,
        "missing": missing,
        "failures": failures,
        "extra": extra,
        "manifest": str(card_dir / "manifest_latest"),
    }
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# V12 All Dev/Offline Qwen3-VL Grounding Audit",
        "",
        f"- generated_at: `{result['generated_at']}`",
        f"- card_dir: `{card_dir}`",
        f"- expected_cards: `{result['expected_count']}`",
        f"- actual_expected_cards: `{result['actual_expected_count']}`",
        f"- ok_cards: `{result['ok_count']}`",
        f"- missing_cards: `{result['missing_count']}`",
        f"- grounding_failed_or_no_top_k: `{result['failed_count']}`",
        f"- extra_cards: `{result['extra_count']}`",
        f"- top_k_total: `{result['top_k_total']}`",
        f"- gpt55_cleaned_count: `{result['gpt55_cleaned_count']}`",
        f"- final_hidden_metadata_used: `no`",
        f"- excluded_final_submission_specs: `{', '.join(result['excluded_final_submission_specs'])}`",
        "",
        "## Status Counts",
        "",
    ]
    for key, value in sorted(status_counts.items()):
        lines.append(f"- `{key}`: `{value}`")
    lines += ["", "## Teacher Counts", ""]
    for key, value in sorted(teacher_counts.items()):
        lines.append(f"- `{key}`: `{value}`")
    lines += ["", "## Parse Status Counts", ""]
    for key, value in sorted(parse_counts.items()):
        lines.append(f"- `{key}`: `{value}`")
    lines += ["", "## Scenario Counts", ""]
    for key, value in sorted(scenario_counts.items()):
        lines.append(f"- `{key}`: `{value}`")
    lines += ["", "## Failed Cards", ""]
    for failure in failures[:200]:
        lines.append(
            f"- `{failure['cache_key']}` scenario=`{failure['scenario']}` status=`{failure['status']}` "
            f"top_k=`{failure['top_k_count']}` parse=`{failure['parse_status']}`"
        )
    if len(failures) > 200:
        lines.append(f"- ... {len(failures) - 200} more failures in JSON report")
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"json": str(report_json), "md": str(report_md), **{k: result[k] for k in ["expected_count", "ok_count", "missing_count", "failed_count", "top_k_total", "gpt55_cleaned_count"]}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
