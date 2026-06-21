#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

CODEX = Path("/home/data-gxu/acm/egolink2026-main/code/track2/codex")
CARD_DIR = CODEX / "visual_cache_v12/qwen3vl_grounding"
MATERIALIZED = CODEX / "state/materialized_splits/validation_A_limit30"
REPORTS = CODEX / "reports"


def expected_keys() -> list[dict]:
    manifest = json.loads((MATERIALIZED / "manifest.json").read_text(encoding="utf-8"))
    rows = []
    for scenario, number, indices in manifest.get("specs", []):
        spec = f"{scenario}{number}"
        for pos, original_idx in enumerate(indices, start=1):
            rows.append({"cache_key": f"{spec}_{pos}", "scenario": scenario, "spec": spec, "task_id": pos, "original_index": int(original_idx)})
    return rows


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    expected = expected_keys()
    expected_by_key = {x["cache_key"]: x for x in expected}
    actual = {}
    for p in sorted(CARD_DIR.glob("*.json")):
        if p.name.startswith("manifest") or p.name.startswith("cleaning_summary"):
            continue
        try:
            card = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            card = {"status": "json_parse_failed", "error": type(exc).__name__, "top_k_candidates": []}
        actual[p.stem] = {"path": str(p), "card": card}

    missing = [x for x in expected if x["cache_key"] not in actual]
    extra = [k for k in actual if k not in expected_by_key]
    failures = []
    status_counts = Counter()
    scenario_counts = Counter()
    topk_total = 0
    teacher_counts = Counter()
    for key, entry in sorted(actual.items()):
        card = entry["card"]
        status = str(card.get("status") or "")
        teacher = str(card.get("teacher") or "")
        topk = card.get("top_k_candidates") or []
        status_counts[status] += 1
        teacher_counts[teacher] += 1
        scenario_counts[str(card.get("scenario") or expected_by_key.get(key, {}).get("scenario") or "unknown")] += 1
        topk_total += len(topk)
        if key in expected_by_key and (status == "grounding_failed" or not topk):
            failures.append(
                {
                    "cache_key": key,
                    "status": status,
                    "teacher": teacher,
                    "top_k_count": len(topk),
                    "original_index": card.get("original_index"),
                    "uncertainty_notes": card.get("uncertainty_notes"),
                    "path": entry["path"],
                }
            )

    ok_count = len(expected) - len(missing) - len(failures)
    ts = time.strftime("%Y%m%d_%H%M%S")
    audit = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "expected_count": len(expected),
        "actual_expected_count": len([k for k in actual if k in expected_by_key]),
        "ok_count": ok_count,
        "missing_count": len(missing),
        "failed_count": len(failures),
        "extra_count": len(extra),
        "top_k_total": topk_total,
        "status_counts": dict(status_counts),
        "teacher_counts": dict(teacher_counts),
        "scenario_counts": dict(scenario_counts),
        "missing": missing,
        "failures": failures,
        "extra": extra,
    }
    json_path = REPORTS / f"V12_VAL41_QWEN3VL_GROUNDING_AUDIT_{ts}.json"
    json_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# V12 Val41 Qwen3VL Grounding Audit",
        "",
        f"- generated_at: `{audit['generated_at']}`",
        f"- expected_cards: `{audit['expected_count']}`",
        f"- actual_expected_cards: `{audit['actual_expected_count']}`",
        f"- ok_cards: `{audit['ok_count']}`",
        f"- missing_cards: `{audit['missing_count']}`",
        f"- grounding_failed_or_no_top_k: `{audit['failed_count']}`",
        f"- extra_cards: `{audit['extra_count']}`",
        f"- top_k_total: `{audit['top_k_total']}`",
        "",
        "## Status Counts",
        "",
    ]
    for k, v in sorted(status_counts.items()):
        lines.append(f"- `{k}`: `{v}`")
    lines += ["", "## Teacher Counts", ""]
    for k, v in sorted(teacher_counts.items()):
        lines.append(f"- `{k}`: `{v}`")
    lines += ["", "## Scenario Counts", ""]
    for k, v in sorted(scenario_counts.items()):
        lines.append(f"- `{k}`: `{v}`")
    if failures:
        lines += ["", "## Failed Cards", ""]
        for f in failures:
            lines.append(f"- `{f['cache_key']}` status=`{f['status']}` top_k=`{f['top_k_count']}` original_index=`{f.get('original_index')}`")
    if missing:
        lines += ["", "## Missing Cards", ""]
        for m in missing:
            lines.append(f"- `{m['cache_key']}` original_index=`{m['original_index']}`")
    md_path = REPORTS / f"V12_VAL41_QWEN3VL_GROUNDING_AUDIT_{ts}.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"json": str(json_path), "md": str(md_path), **{k: audit[k] for k in ["expected_count", "ok_count", "missing_count", "failed_count", "top_k_total"]}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
