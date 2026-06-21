#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract task video frames and contact sheets for Track2 V6."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional


EGO_ROOT = Path(os.environ.get("EGO_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/EgoBench"))
CODEX_ROOT = Path(os.environ.get("CODEX_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/codex"))


def _scenario_path(scenario: str, number: int) -> Path:
    return EGO_ROOT / "scenarios" / "final" / f"{scenario}{number}.json"


def _resolve_video(path_value: str, scenario: str) -> tuple[Optional[Path], str]:
    if not path_value:
        return None, "missing_in_json"
    p = Path(path_value)
    candidates = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(EGO_ROOT / path_value)
        candidates.append(EGO_ROOT / "videos" / path_value)
        candidates.append(EGO_ROOT / "videos" / p.name)
    for cand in candidates:
        if cand.exists():
            return cand, "exists"
    basename = p.name
    videos = EGO_ROOT / "videos"
    if videos.exists():
        matches = list(videos.rglob(basename))
        if not matches and basename:
            stem = p.stem.lower()
            matches = [x for x in videos.rglob("*") if x.is_file() and (stem in x.stem.lower() or scenario.lower() in x.stem.lower())]
        if matches:
            return matches[0], "fuzzy_found"
    return None, "missing_video"


def _duration_ffprobe(video: Path) -> float:
    try:
        out = subprocess.check_output([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=nokey=1:noprint_wrappers=1", str(video)
        ], text=True, timeout=20)
        return max(0.0, float(out.strip()))
    except Exception:
        return 0.0


def _timestamps(duration: float, max_frames: int = 12) -> List[float]:
    points = {0.0}
    if duration > 0:
        points.add(max(0.0, duration / 2.0))
        points.add(max(0.0, duration - 0.2))
        t = 0.0
        while t < duration and len(points) < max_frames:
            points.add(round(t, 2))
            t += 2.0
    out = sorted(points)
    if len(out) > max_frames:
        step = max(1, math.ceil(len(out) / max_frames))
        out = out[::step][:max_frames]
    return out


def _extract_ffmpeg(video: Path, frames_dir: Path, timestamps: List[float]) -> List[Dict[str, Any]]:
    frames = []
    if not shutil.which("ffmpeg"):
        return frames
    for idx, ts in enumerate(timestamps):
        out = frames_dir / f"frame_{idx:02d}_{ts:.2f}s.jpg"
        cmd = ["ffmpeg", "-y", "-ss", f"{ts:.2f}", "-i", str(video), "-frames:v", "1", "-q:v", "3", "-vf", "scale=480:-1", str(out)]
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30, check=True)
            if out.exists() and out.stat().st_size > 0:
                frames.append({"path": str(out), "timestamp": ts})
        except Exception:
            continue
    return frames


def _extract_cv2(video: Path, frames_dir: Path, timestamps: List[float]) -> List[Dict[str, Any]]:
    try:
        import cv2
    except Exception:
        return []
    frames = []
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    for idx, ts in enumerate(timestamps):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(ts * fps))
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        new_w = 480
        new_h = max(1, int(h * new_w / max(1, w)))
        frame = cv2.resize(frame, (new_w, new_h))
        out = frames_dir / f"frame_{idx:02d}_{ts:.2f}s.jpg"
        cv2.imwrite(str(out), frame)
        frames.append({"path": str(out), "timestamp": ts})
    cap.release()
    return frames


def _contact_sheet(frames: List[Dict[str, Any]], out_path: Path) -> bool:
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return False
    if not frames:
        return False
    thumbs = []
    for fr in frames:
        try:
            img = Image.open(fr["path"]).convert("RGB")
            img.thumbnail((320, 180))
            canvas = Image.new("RGB", (320, 210), "white")
            canvas.paste(img, ((320 - img.width) // 2, 0))
            draw = ImageDraw.Draw(canvas)
            draw.text((8, 184), f"{fr['timestamp']:.1f}s", fill=(0, 0, 0))
            thumbs.append(canvas)
        except Exception:
            continue
    if not thumbs:
        return False
    cols = 3 if len(thumbs) > 8 else 2
    rows = math.ceil(len(thumbs) / cols)
    sheet = Image.new("RGB", (cols * 320, rows * 210), "white")
    for i, img in enumerate(thumbs):
        sheet.paste(img, ((i % cols) * 320, (i // cols) * 210))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=85)
    return True


def process_task(scenario: str, number: int, task_index: int, force: bool = False) -> Dict[str, Any]:
    scenario_file = _scenario_path(scenario, number)
    tasks = json.loads(scenario_file.read_text(encoding="utf-8"))
    idx = max(0, task_index - 1)
    task = tasks[idx]
    cache_id = f"{scenario}{number}_{task_index}"
    root = CODEX_ROOT / "visual_cache" / cache_id
    frames_dir = root / "frames"
    root.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "frames_manifest.json"
    if manifest_path.exists() and not force:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_path = task.get("image_path") or task.get("video_path") or task.get("video") or ""
    video, status = _resolve_video(str(raw_path), scenario)
    manifest: Dict[str, Any] = {
        "cache_id": cache_id,
        "scenario": scenario,
        "scenario_number": number,
        "task_index": task_index,
        "raw_path": raw_path,
        "video_path": str(video) if video else "",
        "path_status": status,
        "frames": [],
        "contact_sheet": str(root / "contact_sheet.jpg"),
    }
    if video:
        duration = _duration_ffprobe(video)
        manifest["duration"] = duration
        timestamps = _timestamps(duration, 12)
        frames = _extract_ffmpeg(video, frames_dir, timestamps) or _extract_cv2(video, frames_dir, timestamps)
        manifest["frames"] = frames
        manifest["contact_sheet_created"] = _contact_sheet(frames, root / "contact_sheet.jpg")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--scenario-number", type=int, required=True)
    parser.add_argument("--task-index", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    manifest = process_task(args.scenario, args.scenario_number, args.task_index, args.force)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

