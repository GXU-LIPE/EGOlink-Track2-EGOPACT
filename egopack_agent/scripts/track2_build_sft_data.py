#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a small Track2 tool-use SFT dataset from dev GT and success traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time


CODEX_ROOT = Path("/home/data-gxu/acm/egolink2026-main/code/track2/codex")
EGO_ROOT = Path("/home/data-gxu/acm/egolink2026-main/code/track2/EgoBench")


def iter_json_files(root: Path):
    for p in root.rglob("*.json"):
        if any(piece in str(p).lower() for piece in ("results", "eval_result")):
            continue
        yield p


def find_gt_calls(obj):
    if isinstance(obj, dict):
        for key in ("ground_truth", "Ground_truth", "gt_tool_calls", "tool_calls", "answer"):
            value = obj.get(key)
            if isinstance(value, list) and value and all(isinstance(x, dict) for x in value):
                if any("tool_name" in x for x in value):
                    return value
        for value in obj.values():
            found = find_gt_calls(value)
            if found:
                return found
    if isinstance(obj, list):
        for value in obj:
            found = find_gt_calls(value)
            if found:
                return found
    return None


def load_tasks():
    samples = []
    for p in iter_json_files(EGO_ROOT / "scenarios"):
        try:
            data = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        items = data if isinstance(data, list) else [data]
        scenario_name = p.stem
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            gt = find_gt_calls(item)
            if not gt:
                continue
            instruction = item.get("Instruction") or item.get("instruction") or ""
            visual = item.get("image_description") or ""
            analysis = item.get("analysis") or item.get("Analysis") or ""
            samples.append({
                "id": f"gt::{scenario_name}::{idx+1}",
                "scenario": scenario_name,
                "task_id": str(idx + 1),
                "mode": "dev",
                "source": "gt",
                "messages": [
                    {"role": "system", "content": "You are an EgoBench service agent. Output pure JSON array when calling tools; otherwise short natural language."},
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": json.dumps(gt, ensure_ascii=False, separators=(",", ":"))},
                ],
                "tools_schema": "compact schema available at runtime",
                "visual_context": visual,
                "planner_state": analysis,
                "target": json.dumps(gt, ensure_ascii=False, separators=(",", ":")),
                "compliance_tags": {"no_external_api": True, "no_final_gt": True},
            })
    return samples


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    out_dir = CODEX_ROOT / "train_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = load_tasks()
    random.Random(args.seed).shuffle(samples)
    n_val = max(1, int(len(samples) * args.val_ratio)) if samples else 0
    val = samples[:n_val]
    train = samples[n_val:]
    rejected = []
    for path, rows in ((out_dir / "sft_track2_tooluse_train.jsonl", train), (out_dir / "sft_track2_tooluse_val.jsonl", val)):
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out_dir / "rejected_samples.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rejected) + ("\n" if rejected else ""), encoding="utf-8")
    (out_dir / "teacher_usage_summary.json").write_text(json.dumps({"deepseek_teacher_calls": 0, "enabled": False}, indent=2) + "\n", encoding="utf-8")
    card = [
        "# Track2 SFT Data Card",
        "",
        f"- generated_at: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}",
        f"- train_samples: {len(train)}",
        f"- val_samples: {len(val)}",
        "- sources: official scenario files with explicit `ground_truth` fields; no hidden/no-GT final data",
        "- teacher_corrected_samples: 0 in this build",
        "- target_format: assistant outputs pure JSON array for tool-call turns",
        "- compliance: no external API required to use this dataset; no final GT included",
    ]
    (out_dir / "sft_track2_data_card.md").write_text("\n".join(card) + "\n", encoding="utf-8")
    print(out_dir / "sft_track2_tooluse_train.jsonl")
    print(out_dir / "sft_track2_tooluse_val.jsonl")
    print(f"samples train={len(train)} val={len(val)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
