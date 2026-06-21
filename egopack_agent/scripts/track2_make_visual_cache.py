# -*- coding: utf-8 -*-
"""Create lightweight visual cache from available Track2 videos."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time


EGO_ROOT = Path(os.environ.get("EGO_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/EgoBench"))
CODEX_ROOT = Path(os.environ.get("CODEX_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/codex"))


def run(cmd):
    return subprocess.run(cmd, text=True, encoding="utf-8", errors="replace", stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def duration(video):
    proc = run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(video)])
    try:
        return float(proc.stdout.strip())
    except Exception:
        return 0.0


def extract_ffmpeg(video, out_dir, dur):
    out_dir.mkdir(parents=True, exist_ok=True)
    times = []
    if dur > 0:
        for t in [0, dur / 2, max(dur - 0.5, 0)]:
            times.append(round(t, 2))
        t = 0
        while t < dur and len(times) < 12:
            times.append(round(t, 2))
            t += 2
    else:
        times = [0]
    times = sorted(set(times))[:12]
    frames = []
    for idx, t in enumerate(times):
        out = out_dir / f"frame_{idx:02d}_{str(t).replace('.', '_')}.jpg"
        if out.exists() and out.stat().st_size > 0:
            frames.append(str(out))
            continue
        proc = run(["ffmpeg", "-y", "-ss", str(t), "-i", str(video), "-frames:v", "1", "-q:v", "2", str(out)])
        if proc.returncode == 0 and out.exists():
            frames.append(str(out))
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    videos = sorted((EGO_ROOT / "videos").glob("*"))
    report_lines = [f"# Visual Cache {args.run_id}", "", f"- timestamp: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}"]
    for video in videos:
        if not video.is_file():
            continue
        vid = video.stem.replace(" ", "_")
        root = CODEX_ROOT / "visual_cache" / vid
        state_path = root / "visual_state.json"
        if args.resume and state_path.exists():
            continue
        dur = duration(video) if not args.dry_run else 0.0
        frames = [] if args.dry_run else extract_ffmpeg(video, root / "frames", dur)
        state = {
            "video_id": vid,
            "source_path": str(video),
            "scenario": "",
            "duration": dur,
            "visible_objects": [],
            "object_attributes": [],
            "brand_or_text_if_visible": [],
            "color": [],
            "position": [],
            "temporal_events": [],
            "human_actions": [],
            "pointed_or_held_objects": [],
            "scene_summary": "",
            "uncertainty_notes": "Auto cache extracted frames only; no stable VLM caption generated in this pass.",
            "evidence_frames": frames,
        }
        if not args.dry_run:
            root.mkdir(parents=True, exist_ok=True)
            with open(state_path, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
            with open(root / "visual_state.txt", "w", encoding="utf-8") as f:
                f.write(json.dumps(state, ensure_ascii=False, indent=2))
        report_lines.append(f"- {video.name}: frames={len(frames)} duration={dur}")
    report = CODEX_ROOT / "reports" / f"visual_cache_{args.run_id or 'latest'}.md"
    if not args.dry_run:
        report.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print("\n".join(report_lines))


if __name__ == "__main__":
    main()
