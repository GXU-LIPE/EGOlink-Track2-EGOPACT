#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build cached V25-new contact sheets for val41 tasks."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont


CODEX = Path(os.environ.get("CODEX_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/codex"))
EGO = Path(os.environ.get("EGO_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/EgoBench"))
SPLIT_DIR = CODEX / "state" / "materialized_splits" / "validation_A_limit30"
OUT_DIR = CODEX / "visual_cache_v25_new" / "contact_sheets"


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return default


def load_specs() -> List[Tuple[str, int, List[int]]]:
    m = read_json(SPLIT_DIR / "manifest.json", {})
    return [(str(s), int(n), [int(x) for x in idxs]) for s, n, idxs in m.get("specs", [])]


def find_video_path(row: Dict[str, Any]) -> str:
    for key in ("video_path", "image_path", "image_name", "video", "image"):
        val = row.get(key)
        if not val:
            continue
        p = Path(str(val))
        if p.exists():
            return str(p)
        base = p.name
        cand = EGO / "videos" / base
        if cand.exists():
            return str(cand)
        if not base.endswith(".mp4"):
            cand = EGO / "videos" / (base + ".mp4")
            if cand.exists():
                return str(cand)
    return ""


def duration_seconds(video: str) -> float:
    try:
        out = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                video,
            ],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=20,
        ).strip()
        return max(1.0, float(out))
    except Exception:
        return 24.0


def sample_times(duration: float, max_frames: int = 12) -> List[float]:
    if duration <= 0:
        return [0.0]
    base = [0.5, max(0.5, duration / 2.0), max(0.5, duration - 0.8)]
    step = 2.0
    t = 0.5
    while t < duration - 0.3:
        base.append(t)
        t += step
    uniq: List[float] = []
    for x in sorted(base):
        y = round(min(max(0.0, x), max(0.0, duration - 0.2)), 2)
        if y not in uniq:
            uniq.append(y)
    if len(uniq) > max_frames:
        idxs = [round(i * (len(uniq) - 1) / (max_frames - 1)) for i in range(max_frames)]
        uniq = [uniq[i] for i in idxs]
    return uniq


def extract_frame(video: str, t: float, out: Path) -> bool:
    try:
        subprocess.check_call(
            [
                "ffmpeg",
                "-y",
                "-ss",
                str(t),
                "-i",
                video,
                "-frames:v",
                "1",
                "-vf",
                "scale=360:-1",
                str(out),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
        )
        return out.exists() and out.stat().st_size > 0
    except Exception:
        return False


def build_sheet(video: str, out_path: Path, max_frames: int = 12) -> Dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        return {"status": "cached", "contact_sheet_path": str(out_path)}
    dur = duration_seconds(video)
    times = sample_times(dur, max_frames=max_frames)
    frames: List[Tuple[float, Image.Image]] = []
    with tempfile.TemporaryDirectory(prefix="v25_frames_") as td:
        tdir = Path(td)
        for i, t in enumerate(times):
            fp = tdir / f"frame_{i:02d}.jpg"
            if not extract_frame(video, t, fp):
                continue
            try:
                img = Image.open(fp).convert("RGB")
                frames.append((t, img))
            except Exception:
                pass
    if not frames:
        return {"status": "failed", "reason": "no_frames", "video": video}
    cols = 4
    rows = (len(frames) + cols - 1) // cols
    thumb_w, thumb_h = 360, 220
    sheet = Image.new("RGB", (cols * thumb_w, rows * thumb_h), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for i, (t, img) in enumerate(frames):
        img.thumbnail((thumb_w, thumb_h - 22))
        x = (i % cols) * thumb_w
        y = (i // cols) * thumb_h
        sheet.paste(img, (x, y + 22))
        draw.rectangle([x, y, x + thumb_w - 1, y + thumb_h - 1], outline=(180, 180, 180), width=1)
        draw.text((x + 6, y + 5), f"{i+1}  t={t:.1f}s", fill=(0, 0, 0), font=font)
    sheet.save(out_path, "JPEG", quality=82)
    return {"status": "built", "contact_sheet_path": str(out_path), "frame_count": len(frames), "duration": dur}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="")
    ap.add_argument("--pos", type=int, default=-1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    rows_out: List[Dict[str, Any]] = []
    count = 0
    for scenario, number, _ in load_specs():
        spec = f"{scenario}{number}"
        if args.spec and spec != args.spec:
            continue
        rows = read_json(SPLIT_DIR / f"{spec}.json", [])
        for pos, row in enumerate(rows):
            if args.pos >= 0 and pos != args.pos:
                continue
            video = find_video_path(row)
            out = OUT_DIR / f"{spec}_{pos + 1}.jpg"
            if not video:
                rec = {"spec": spec, "local_pos": pos, "status": "missing_video"}
            else:
                rec = {"spec": spec, "local_pos": pos, "video": video, **build_sheet(video, out)}
            rows_out.append(rec)
            count += 1
            if args.limit and count >= args.limit:
                break
        if args.limit and count >= args.limit:
            break
    if not args.quiet:
        print(json.dumps(rows_out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
