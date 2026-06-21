#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

CODEX = Path("/home/data-gxu/acm/egolink2026-main/code/track2/codex")
EGO = Path("/home/data-gxu/acm/egolink2026-main/code/track2/EgoBench")
CODE1 = Path("/home/data-gxu/acm/egolink2026-main/code1")
DEFAULT_MODEL = CODE1 / "models/Qwen/Qwen3-VL-30B-A3B-Instruct"
DEFAULT_OUT = CODEX / "visual_cache_v12/qwen3vl_grounding_all_dev_offline"
VIDEO_OUT = CODEX / "visual_cache_v12/qwen3vl_grounding_by_video"
FRAME_CACHE = CODEX / "visual_cache_v12/qwen3vl_frames_all_by_video"

FINAL_SUBMISSION_SPECS = {"retail6", "retail10", "kitchen4", "restaurant5", "order2"}
SCENARIOS = ("retail", "kitchen", "restaurant", "order")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def scenario_from_spec(spec: str) -> str:
    for scenario in SCENARIOS:
        if spec.startswith(scenario):
            return scenario
    return "unknown"


def resolve_video(raw: str) -> Path:
    raw = str(raw or "")
    p = Path(raw)
    candidates: list[Path] = []
    if p.is_absolute():
        candidates.append(p)
    candidates.extend([EGO / "videos" / raw, EGO / "videos" / Path(raw).name])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    videos = EGO / "videos"
    low = raw.lower()
    if low == "order1.mp4" or ("greek" in low and "annie" in low):
        mapped = videos / "greek_annie_1.mp4"
        if mapped.exists():
            return mapped
    stem = Path(raw).stem.lower()
    if stem:
        fuzzy = sorted(videos.glob(f"*{stem}*.mp4"))
        if fuzzy:
            return fuzzy[0]
    return candidates[-1] if candidates else p


def collect_nonfinal_tasks() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tasks: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for root in [EGO / "scenarios" / "final", EGO / "scenarios"]:
        if not root.exists():
            continue
        for path in sorted(root.glob("*.json")):
            spec = path.stem.lower()
            data = read_json(path)
            if not isinstance(data, list):
                continue
            if spec in FINAL_SUBMISSION_SPECS:
                excluded.append({"file": str(path), "reason": "official_final_submission_spec_excluded", "tasks": len(data)})
                continue
            for idx, row in enumerate(data, start=1):
                key = (spec, idx)
                if key in seen or not isinstance(row, dict):
                    continue
                seen.add(key)
                instruction = str(row.get("Instruction") or row.get("instruction") or "")
                raw_video = row.get("image_path") or row.get("video_path") or row.get("video") or ""
                tasks.append(
                    {
                        "spec": spec,
                        "scenario": scenario_from_spec(spec),
                        "task_id": idx,
                        "instruction_digest": instruction[:500],
                        "video_path": resolve_video(raw_video),
                        "source_file": str(path),
                        "source_split": "dev_offline_nonfinal",
                    }
                )
    return tasks, excluded


def fallback_card(scenario: str, video_path: Path, status: str, error: str = "") -> dict[str, Any]:
    return {
        "status": status,
        "teacher": "qwen3vl",
        "error": error,
        "scenario": scenario,
        "video_path": str(video_path),
        "path_status": "exists" if video_path.exists() else "missing_video",
        "scene_summary": "",
        "visible_text": [],
        "visible_products": [],
        "visible_dishes": [],
        "visible_ingredients": [],
        "pointed_or_held_objects": [],
        "relative_location_objects": [],
        "category_country_brand_taste_clues": [],
        "restaurant_menu_order_clues": [],
        "top_k_candidates": [],
        "uncertainty_notes": [error] if error else [],
        "final_hidden_metadata_used": False,
        "_qwen3vl_json_parse_status": "not_run",
    }


def build_video_prompt(scenario: str, specs: list[str], task_count: int) -> str:
    return f"""You are a visual grounding teacher for EgoBench Track2. Inspect sampled first-person video frames and return one strict JSON object only.
Do not call tools. Do not solve database tasks. Do not invent hidden answers.
This video is reused by {task_count} non-final dev/offline tasks for scenario={scenario}, specs={sorted(set(specs))}.

Identify all visually grounded candidate entities that may help those tasks. Include plausible low-confidence candidates when visual evidence exists.
Return compact JSON with exactly these keys:
{{
  "scene_summary": "...",
  "visible_text": ["..."],
  "visible_products": ["..."],
  "visible_dishes": ["..."],
  "visible_ingredients": ["..."],
  "pointed_or_held_objects": ["..."],
  "relative_location_objects": ["..."],
  "category_country_brand_taste_clues": ["..."],
  "restaurant_menu_order_clues": ["..."],
  "top_k_candidates": [{{"entity":"...","type":"product|dish|set_meal|ingredient|restaurant|category|object","evidence":"...","confidence":0.0}}],
  "uncertainty_notes": ["..."]
}}

Scenario focus:
- retail: product labels, brand, category, country/origin, taste/profile clues, price labels, shelf position, pointed/held items.
- restaurant/order: restaurant/menu text, dish names, set meal names, category headings, current order, pointed/replaced/removed item.
- kitchen: current recipe step, visible ingredients, cooking tools/containers, fridge/stove/tray, action sequence.
"""


def parse_qwen3vl_json(raw: str) -> tuple[dict[str, Any], str]:
    raw = (raw or "").strip()
    if not raw:
        return {"scene_summary": "", "uncertainty_notes": ["empty qwen3vl output"]}, "empty"
    candidate = raw
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, flags=re.S)
    if fenced:
        candidate = fenced.group(1).strip()
    else:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            candidate = raw[start : end + 1]
    try:
        return json.loads(candidate), "parsed_json"
    except Exception:
        repaired = candidate.replace("“", '"').replace("”", '"').replace("’", "'")
        repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
        try:
            return json.loads(repaired), "repaired_json"
        except Exception:
            return {"scene_summary": raw[:2500], "uncertainty_notes": ["Qwen3-VL returned malformed JSON; raw text retained for GPT-5.5 cleanup."]}, "malformed_json_raw_scene"


def load_qwen3vl(model_path: Path):
    sys.path.insert(0, str(CODE1))
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        str(model_path),
        torch_dtype=dtype,
        device_map="auto",
        quantization_config=quant,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
    return model, processor


def ground_video(video_path: Path, scenario: str, specs: list[str], task_count: int, model: Any, processor: Any, frame_count: int, max_new_tokens: int) -> dict[str, Any]:
    if not video_path.exists():
        return fallback_card(scenario, video_path, "grounding_failed", f"video_missing:{video_path}")
    try:
        sys.path.insert(0, str(CODE1))
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        import torch
        from egolink_code1.video import sample_video_frames
        from qwen_vl_utils import process_vision_info
    except Exception as exc:
        return fallback_card(scenario, video_path, "grounding_failed", f"import_error:{type(exc).__name__}:{str(exc)[:300]}")
    try:
        cache_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", video_path.stem)
        frames = sample_video_frames(video_path, FRAME_CACHE / cache_key, frame_count=frame_count)
        content = [{"type": "image", "image": str(frame), "max_pixels": 230400} for frame in frames]
        content.append({"type": "text", "text": build_video_prompt(scenario, specs, task_count)})
        messages = [{"role": "user", "content": content}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info(messages)
        inputs = processor(text=[text], images=images, videos=videos or None, padding=True, return_tensors="pt")
        input_len = inputs["input_ids"].shape[1]
        device = next(model.parameters()).device
        for key, value in list(inputs.items()):
            if torch.is_tensor(value):
                inputs[key] = value.to(device)
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        raw = processor.batch_decode(output[:, input_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        parsed, parse_status = parse_qwen3vl_json(raw)
        card = {**fallback_card(scenario, video_path, "qwen3vl_success"), **parsed}
        card.update(
            {
                "teacher": "qwen3vl",
                "status": "qwen3vl_success",
                "scenario": scenario,
                "video_path": str(video_path),
                "source_split": "dev_offline_nonfinal",
                "task_count_for_video": task_count,
                "scenario_specs_for_video": sorted(set(specs)),
                "raw_output_preview": raw[:8000],
                "final_hidden_metadata_used": False,
                "_qwen3vl_json_parse_status": parse_status,
            }
        )
        return card
    except Exception as exc:
        return fallback_card(scenario, video_path, "grounding_failed", f"inference_error:{type(exc).__name__}:{str(exc)[:500]}")


def backup_existing_cache(out_dir: Path, backup_root: Path) -> Path | None:
    if not out_dir.exists() or not any(out_dir.glob("*.json")):
        return None
    backup_root.mkdir(parents=True, exist_ok=True)
    backup = backup_root / f"{out_dir.name}_before_by_video_{time.strftime('%Y%m%d_%H%M%S')}"
    if backup.exists():
        raise SystemExit(f"backup path already exists: {backup}")
    shutil.copytree(out_dir, backup)
    return backup


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def materialize_task_card(task: dict[str, Any], video_card: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    card = dict(video_card)
    card.update(
        {
            "scenario": task["scenario"],
            "scenario_spec": task["spec"],
            "task_id": task["task_id"],
            "cache_key": f"{task['spec']}_{task['task_id']}",
            "instruction_digest": task.get("instruction_digest", ""),
            "source_file": task.get("source_file", ""),
            "source_split": task.get("source_split", "dev_offline_nonfinal"),
            "video_grounding_source": "qwen3vl_by_unique_video",
            "final_hidden_metadata_used": False,
        }
    )
    out = out_dir / f"{task['spec']}_{task['task_id']}.json"
    write_json(out, card)
    return {
        "cache_key": card["cache_key"],
        "path": str(out),
        "scenario": task["scenario"],
        "spec": task["spec"],
        "task_id": task["task_id"],
        "status": card.get("status"),
        "teacher": card.get("teacher"),
        "parse_status": card.get("_qwen3vl_json_parse_status"),
        "video": str(task["video_path"]),
        "video_exists": task["video_path"].exists(),
        "top_k_count": len(card.get("top_k_candidates") or []),
        "sha256": hashlib.sha256(out.read_bytes()).hexdigest() if out.exists() else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT))
    parser.add_argument("--video_out_dir", default=str(VIDEO_OUT))
    parser.add_argument("--model_path", default=str(DEFAULT_MODEL))
    parser.add_argument("--frame_count", type=int, default=12)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--clear_out_dir", action="store_true")
    parser.add_argument("--skip_existing_video", action="store_true")
    parser.add_argument("--no_backup", action="store_true")
    parser.add_argument("--require_qwen3vl", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    video_out_dir = Path(args.video_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    video_out_dir.mkdir(parents=True, exist_ok=True)
    backup = None
    if not args.no_backup:
        backup = backup_existing_cache(out_dir, out_dir.parent / "backups")
    if args.clear_out_dir:
        for old in out_dir.glob("*.json"):
            old.unlink()

    tasks, excluded = collect_nonfinal_tasks()
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        by_video[str(task["video_path"])].append(task)

    model_path = Path(args.model_path)
    model = None
    processor = None
    model_error = ""
    if not model_path.exists():
        model_error = f"model_path_missing:{model_path}"
    else:
        try:
            model, processor = load_qwen3vl(model_path)
        except Exception as exc:
            model_error = f"{type(exc).__name__}:{str(exc)[:500]}"
    if args.require_qwen3vl and (model is None or processor is None):
        raise SystemExit(f"Qwen3-VL model load failed: {model_error}")

    video_items: list[dict[str, Any]] = []
    video_cards: dict[str, dict[str, Any]] = {}
    for video_str, group in sorted(by_video.items()):
        video_path = Path(video_str)
        scenario = group[0]["scenario"]
        specs = [task["spec"] for task in group]
        video_key = hashlib.sha1(video_str.encode("utf-8")).hexdigest()[:16] + "_" + re.sub(r"[^A-Za-z0-9_.-]+", "_", video_path.stem)[:80]
        video_card_path = video_out_dir / f"{video_key}.json"
        if args.skip_existing_video and video_card_path.exists():
            card = read_json(video_card_path)
        elif model is None or processor is None:
            card = fallback_card(scenario, video_path, "grounding_failed", f"model_unavailable:{model_error}")
            write_json(video_card_path, card)
        else:
            card = ground_video(video_path, scenario, specs, len(group), model, processor, args.frame_count, args.max_new_tokens)
            write_json(video_card_path, card)
        video_cards[video_str] = card
        item = {
            "video": video_str,
            "video_card": str(video_card_path),
            "task_count": len(group),
            "scenario": scenario,
            "specs": sorted(set(specs)),
            "status": card.get("status"),
            "teacher": card.get("teacher"),
            "parse_status": card.get("_qwen3vl_json_parse_status"),
            "top_k_count": len(card.get("top_k_candidates") or []),
            "sha256": hashlib.sha256(video_card_path.read_bytes()).hexdigest(),
        }
        video_items.append(item)
        print(json.dumps({"video_grounding": item}, ensure_ascii=False), flush=True)

    manifest_items: list[dict[str, Any]] = []
    for task in tasks:
        manifest_items.append(materialize_task_card(task, video_cards[str(task["video_path"])], out_dir))
        print(json.dumps(manifest_items[-1], ensure_ascii=False), flush=True)

    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_split": "dev_offline_nonfinal",
        "task_count": len(tasks),
        "unique_video_count": len(by_video),
        "excluded_final_submission_specs": sorted(FINAL_SUBMISSION_SPECS),
        "excluded_files": excluded,
        "out_dir": str(out_dir),
        "video_out_dir": str(video_out_dir),
        "backup": str(backup) if backup else "",
        "model_path": str(model_path),
        "model_loaded": model is not None and processor is not None,
        "model_error": model_error,
        "final_hidden_metadata_used": False,
        "video_items": video_items,
        "items": manifest_items,
    }
    manifest_path = out_dir / f"manifest_all_dev_offline_{time.strftime('%Y%m%d_%H%M%S')}.json"
    write_json(manifest_path, manifest)
    print("manifest", manifest_path, flush=True)


if __name__ == "__main__":
    main()
