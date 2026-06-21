#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

CODEX = Path("/home/data-gxu/acm/egolink2026-main/code/track2/codex")
if str(CODEX / "scripts") not in sys.path:
    sys.path.insert(0, str(CODEX / "scripts"))

from build_v12_qwen3vl_grounding import (  # noqa: E402
    DEFAULT_MODEL,
    fallback_card,
    load_qwen3vl,
    resolve_video,
    try_qwen3vl_ground,
)


def load_materialized_tasks(materialized_dir: Path) -> list[dict[str, Any]]:
    manifest_path = materialized_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"missing materialized split manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tasks: list[dict[str, Any]] = []
    for scenario, number, indices in manifest.get("specs", []):
        spec = f"{scenario}{number}"
        scenario_path = materialized_dir / f"{spec}.json"
        if not scenario_path.exists():
            raise SystemExit(f"missing materialized scenario json: {scenario_path}")
        rows = json.loads(scenario_path.read_text(encoding="utf-8"))
        if len(rows) != len(indices):
            print(
                json.dumps(
                    {
                        "warning": "row_count_mismatch",
                        "spec": spec,
                        "rows": len(rows),
                        "manifest_indices": len(indices),
                    },
                    ensure_ascii=False,
                )
            )
        for subset_pos, row in enumerate(rows, start=1):
            original_index = int(row.get("_v8_original_index") or indices[subset_pos - 1])
            raw_video = row.get("image_path") or row.get("video_path") or row.get("video") or ""
            task = {
                "spec": spec,
                "scenario": str(scenario),
                "number": int(number),
                "task_id": subset_pos,
                "original_index": original_index,
                "row": row,
                "video_path": resolve_video(raw_video),
            }
            tasks.append(task)
    return tasks


def backup_existing_cache(out_dir: Path, backup_root: Path) -> Path | None:
    if not out_dir.exists() or not any(out_dir.glob("*.json")):
        return None
    backup_root.mkdir(parents=True, exist_ok=True)
    backup = backup_root / f"{out_dir.name}_before_val41_allcards_{time.strftime('%Y%m%d_%H%M%S')}"
    if backup.exists():
        raise SystemExit(f"backup path already exists: {backup}")
    shutil.copytree(out_dir, backup)
    return backup


def write_card(out_dir: Path, task: dict[str, Any], card: dict[str, Any]) -> Path:
    card["scenario_spec"] = task["spec"]
    card["scenario"] = task["scenario"]
    card["task_id"] = task["task_id"]
    card["original_index"] = task["original_index"]
    card["cache_key"] = f"{task['spec']}_{task['task_id']}"
    card["final_hidden_metadata_used"] = False
    card["source_split"] = "validation_A_medium"
    out = out_dir / f"{task['spec']}_{task['task_id']}.json"
    out.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--materialized_dir",
        default=str(CODEX / "state/materialized_splits/validation_A_limit30"),
    )
    parser.add_argument(
        "--out_dir",
        default=str(CODEX / "visual_cache_v12/qwen3vl_grounding"),
    )
    parser.add_argument("--model_path", default=str(DEFAULT_MODEL))
    parser.add_argument("--frame_count", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=768)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--no_backup", action="store_true")
    parser.add_argument("--clear_out_dir", action="store_true")
    parser.add_argument("--fallback_only", action="store_true")
    parser.add_argument("--require_qwen3vl", action="store_true")
    args = parser.parse_args()

    materialized_dir = Path(args.materialized_dir)
    out_dir = Path(args.out_dir)
    backup_root = out_dir.parent / "backups"
    out_dir.mkdir(parents=True, exist_ok=True)

    backup = None
    if not args.no_backup:
        backup = backup_existing_cache(out_dir, backup_root)
    if args.clear_out_dir:
        for p in out_dir.glob("*.json"):
            p.unlink()

    tasks = load_materialized_tasks(materialized_dir)
    model = None
    processor = None
    model_error = ""
    model_path = Path(args.model_path)
    if not args.fallback_only:
        if not model_path.exists():
            model_error = f"model_path_missing:{model_path}"
        else:
            try:
                model, processor = load_qwen3vl(model_path)
            except Exception as exc:
                model_error = type(exc).__name__ + ":" + str(exc)[:500]
    if args.require_qwen3vl and (model is None or processor is None):
        raise SystemExit(f"Qwen3-VL model load failed in require mode: {model_error}")

    manifest_items = []
    for task in tasks:
        out = out_dir / f"{task['spec']}_{task['task_id']}.json"
        if args.skip_existing and out.exists():
            card = json.loads(out.read_text(encoding="utf-8"))
            status = card.get("status", "existing")
        elif args.fallback_only:
            card = fallback_card(task, "fallback_forced")
            out = write_card(out_dir, task, card)
            status = card.get("status")
        elif model is None or processor is None:
            status_name = "fallback_model_missing" if model_error.startswith("model_path_missing") else "fallback_model_load_error"
            card = fallback_card(task, status_name, model_error)
            out = write_card(out_dir, task, card)
            status = card.get("status")
        else:
            card = try_qwen3vl_ground(task, model, processor, args.frame_count, args.max_new_tokens)
            if args.require_qwen3vl and card.get("status") != "qwen3vl_success":
                write_card(out_dir, task, card)
                raise SystemExit(f"Qwen3-VL grounding failed for {task['spec']}_{task['task_id']}: {card.get('status')} {card.get('error','')}")
            out = write_card(out_dir, task, card)
            status = card.get("status")

        item = {
            "cache_key": f"{task['spec']}_{task['task_id']}",
            "path": str(out),
            "scenario": task["scenario"],
            "spec": task["spec"],
            "task_id": task["task_id"],
            "original_index": task["original_index"],
            "status": status,
            "teacher": card.get("teacher"),
            "video": str(task["video_path"]),
            "video_exists": task["video_path"].exists(),
            "top_k_count": len(card.get("top_k_candidates") or []),
            "sha256": hashlib.sha256(out.read_bytes()).hexdigest() if out.exists() else "",
        }
        manifest_items.append(item)
        print(json.dumps(item, ensure_ascii=False), flush=True)

    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "split": "validation_A_medium",
        "task_count": len(tasks),
        "out_dir": str(out_dir),
        "backup": str(backup) if backup else "",
        "model_path": str(model_path),
        "model_loaded": model is not None and processor is not None,
        "model_error": model_error,
        "items": manifest_items,
    }
    manifest_path = out_dir / f"manifest_val41_{time.strftime('%Y%m%d_%H%M%S')}.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("manifest", manifest_path, flush=True)


if __name__ == "__main__":
    main()
