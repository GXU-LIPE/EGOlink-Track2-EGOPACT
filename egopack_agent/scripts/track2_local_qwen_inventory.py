#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Find local Qwen2.5-VL weights and write an inventory report."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time


CODEX_ROOT = Path("/home/data-gxu/acm/egolink2026-main/code/track2/codex")
DEFAULT_ROOTS = [
    "/home/data-gxu",
    "/home/data-gxu/models",
    "/home/data-gxu/acm",
    "/data",
    "/workspace",
    "/root/.cache/huggingface",
]
KEYWORDS = ["Qwen2.5-VL-32B-Instruct", "qwen2.5-vl-32b", "Qwen2.5-VL", "Qwen2-VL", "32B-Instruct"]


def dir_size(path: Path) -> int:
    total = 0
    try:
        for p in path.rglob("*"):
            try:
                if p.is_file():
                    total += p.stat().st_size
            except OSError:
                pass
    except OSError:
        pass
    return total


def inspect_candidate(path: Path) -> dict:
    files = {p.name for p in path.iterdir()} if path.is_dir() else set()
    safetensors = list(path.glob("*.safetensors")) if path.is_dir() else []
    info = {
        "path": str(path),
        "size_bytes": dir_size(path),
        "has_config_json": "config.json" in files,
        "has_tokenizer": any(name.startswith("tokenizer") for name in files),
        "has_processor": any("processor" in name.lower() for name in files),
        "safetensors_count": len(safetensors),
        "recommended": False,
    }
    info["recommended"] = info["has_config_json"] and info["safetensors_count"] > 0
    return info


def find_candidates(roots: list[str]) -> list[Path]:
    out: list[Path] = []
    seen = set()
    for root_s in roots:
        root = Path(os.path.expanduser(root_s))
        try:
            if not root.exists():
                continue
        except OSError:
            continue
        root_depth = len(root.parts)

        def _onerror(_exc):
            return None

        for dirpath, dirnames, filenames in os.walk(root, onerror=_onerror):
            path = Path(dirpath)
            if len(path.parts) - root_depth > 8:
                dirnames[:] = []
                continue
            dirnames[:] = [
                d for d in dirnames[:80]
                if d not in {".git", "__pycache__", "node_modules", "wandb", "logs", "eval_result", "results"}
            ]
            name = path.name
            joined = str(path)
            if any(k.lower() in joined.lower() or k.lower() in name.lower() for k in KEYWORDS):
                if str(path) not in seen:
                    seen.add(str(path))
                    out.append(path)
            if len(out) >= 50:
                return out
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="*", default=DEFAULT_ROOTS)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    ts = time.strftime("%Y%m%d_%H%M%S")
    report = Path(args.output) if args.output else CODEX_ROOT / "reports" / f"04_local_qwen_inventory_{ts}.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    candidates = [inspect_candidate(p) for p in find_candidates(args.roots)]
    recommended = next((c for c in candidates if c["recommended"] and "32" in c["path"].lower()), None)
    backend = "local_qwen_vllm" if recommended else "blocked_no_local_weights"
    lines = [
        "# Local Qwen Inventory",
        "",
        f"- generated_at: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}",
        f"- recommended_backend: {backend}",
        f"- recommended_path: {recommended['path'] if recommended else 'NONE'}",
        "",
        "## Candidates",
        "",
    ]
    if not candidates:
        lines.append("No local Qwen/Qwen2.5-VL candidate directories found in configured roots.")
    for c in candidates:
        lines += [
            f"### {c['path']}",
            f"- size_gb: {c['size_bytes'] / (1024**3):.2f}",
            f"- config.json: {c['has_config_json']}",
            f"- tokenizer files: {c['has_tokenizer']}",
            f"- processor files: {c['has_processor']}",
            f"- safetensors_count: {c['safetensors_count']}",
            f"- recommended: {c['recommended']}",
            "",
        ]
    state = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "recommended_backend": backend,
        "recommended_path": recommended["path"] if recommended else "",
        "candidates": candidates,
    }
    (CODEX_ROOT / "state" / "local_qwen_inventory.json").write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
