#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Distill V16 GT100 samples into strict JSON rule records with GPT-5.5.

The script is API-safe: it reads keys from codex/state/.openai_env, never
prints them, writes shard files atomically, and falls back to deterministic
records for failed calls. It does not use final or val41 labels.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import random
import re
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional


CODEX = Path(os.environ.get("CODEX_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/codex"))
OUT_DIR = CODEX / "gt_distill_v16"


def load_shell_env(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line.startswith("export ") or "=" not in line:
            continue
        key, val = line[len("export "):].split("=", 1)
        out[key.strip()] = val.strip().strip('"').strip("'")
    return out


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    tmp.replace(path)


def deterministic_rule(sample: Dict[str, Any], error: str = "") -> Dict[str, Any]:
    names = sample.get("tool_names") or []
    slots = sample.get("entity_slots") or {}
    return {
        "pool_id": sample.get("pool_id"),
        "task_type": sample.get("task_type"),
        "minimal_tool_skeleton": names,
        "required_slots": {k: "canonicalize_before_tool" for k in slots},
        "branch_logic": {"requires_branch_check": any(x in (sample.get("instruction", "") + sample.get("analysis", "")).lower() for x in ["if ", "otherwise", "else", "whether"])},
        "canonical_entities": slots,
        "state_pins": {k: slots.get(k) for k in ("user_id", "restaurant_name") if k in slots},
        "mutation_closure": [n for n in names if re.search(r"^(add|remove|delete|update|modify)_|_(to|from)_", str(n))],
        "aggregate_closure": [n for n in names if str(n).startswith(("compute_total_", "tally_total_"))],
        "anti_patterns": ["broad_scan", "missing_canonicalization", "missing_aggregate_closure"],
        "generalizable_rule": f"For {sample.get('scenario')}/{sample.get('task_type')}, follow the observed minimal skeleton and fill slots via retrieval before mutation.",
        "source": "deterministic_fallback" if error else "deterministic",
        "api_error": error,
        "no_final_metadata": True,
        "uses_val41_label": False,
    }


def build_prompt(sample: Dict[str, Any]) -> str:
    compact = {
        "scenario": sample.get("scenario"),
        "spec": sample.get("spec"),
        "task_type": sample.get("task_type"),
        "instruction": (sample.get("instruction") or "")[:1200],
        "analysis": (sample.get("analysis") or "")[:1200],
        "tool_names": sample.get("tool_names"),
        "entity_slots": sample.get("entity_slots"),
        "tool_calls": sample.get("tool_calls"),
    }
    return (
        "You distill EgoBench Track2 non-final, non-val41 GT tool trajectories into executable generalized rules. "
        "Do not copy final answers. Output JSON only with keys: task_type, minimal_tool_skeleton, required_slots, "
        "branch_logic, canonical_entities, state_pins, mutation_closure, aggregate_closure, anti_patterns, generalizable_rule.\n\n"
        + json.dumps(compact, ensure_ascii=False)
    )


def call_api(sample: Dict[str, Any], env: Dict[str, str], timeout: int) -> Dict[str, Any]:
    key = env.get("OPENAI_API_KEY") or env.get("SERVICE_MODEL_API_KEY")
    base = env.get("TRACK2_OPENAI_BASE_URL") or env.get("SERVICE_MODEL_API_BASE") or "https://ai-pixel.online/v1"
    model = env.get("TRACK2_OPENAI_MODEL") or env.get("SERVICE_MODEL_NAME") or "gpt-5.5"
    if not key:
        return deterministic_rule(sample, "missing_api_key")
    url = base.rstrip("/")
    if not url.endswith("/chat/completions"):
        url += "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return strict JSON only."},
            {"role": "user", "content": build_prompt(sample)},
        ],
        "temperature": 0.0,
        "max_tokens": 900,
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        method="POST",
    )
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), urllib.request.HTTPSHandler(context=ssl.create_default_context()))
        with opener.open(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8", errors="replace"))
        text = body.get("choices", [{}])[0].get("message", {}).get("content", "")
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I | re.S)
        obj = json.loads(text)
        obj.update({
            "pool_id": sample.get("pool_id"),
            "source": "gpt55",
            "api_model": model,
            "no_final_metadata": True,
            "uses_val41_label": False,
        })
        return obj
    except Exception as exc:
        return deterministic_rule(sample, f"{type(exc).__name__}: {str(exc)[:180]}")


def worker(batch: List[Dict[str, Any]], env: Dict[str, str], timeout: int, shard_path: Path) -> Dict[str, Any]:
    rows = []
    errors = 0
    for sample in batch:
        row = call_api(sample, env, timeout)
        if row.get("api_error"):
            errors += 1
        rows.append(row)
    append_jsonl(shard_path, rows)
    return {"shard": str(shard_path), "rows": len(rows), "errors": errors}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default=f"v16_gpt55_distill_{time.strftime('%Y%m%d_%H%M%S')}")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--max-samples", type=int, default=120)
    ap.add_argument("--timeout", type=int, default=120)
    args = ap.parse_args()

    pool = read_jsonl(OUT_DIR / "gt100_pool.jsonl")
    # Representative deterministic ordering: high-value bottleneck scenarios first, then stable sample.
    priority = {"order": 0, "retail": 1, "restaurant": 2, "kitchen": 3}
    pool = sorted(pool, key=lambda r: (priority.get(r.get("scenario"), 9), r.get("task_type", ""), r.get("spec", ""), str(r.get("task_id"))))
    if args.max_samples and len(pool) > args.max_samples:
        head = pool[: args.max_samples // 2]
        rest = pool[args.max_samples // 2 :]
        random.Random(16).shuffle(rest)
        pool = head + rest[: args.max_samples - len(head)]

    env = os.environ.copy()
    env.update(load_shell_env(CODEX / "state" / ".openai_env"))
    shard_dir = OUT_DIR / "gpt55_rule_shards" / args.run_id
    shard_dir.mkdir(parents=True, exist_ok=True)
    shards: List[List[Dict[str, Any]]] = [[] for _ in range(max(1, args.workers))]
    for i, sample in enumerate(pool):
        shards[i % len(shards)].append(sample)

    summaries = []
    with cf.ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = []
        for i, batch in enumerate(shards):
            if batch:
                futs.append(ex.submit(worker, batch, env, args.timeout, shard_dir / f"shard_{i:02d}.jsonl"))
        for fut in cf.as_completed(futs):
            item = fut.result()
            summaries.append(item)
            print(json.dumps(item, ensure_ascii=False), flush=True)

    merged = []
    for shard in sorted(shard_dir.glob("shard_*.jsonl")):
        merged.extend(read_jsonl(shard))
    out_path = OUT_DIR / "gpt55_rule_distillation.jsonl"
    append_jsonl(out_path, merged)
    summary = {
        "run_id": args.run_id,
        "pool_available": len(read_jsonl(OUT_DIR / "gt100_pool.jsonl")),
        "samples_requested": len(pool),
        "samples_completed": len(merged),
        "gpt55_rows": sum(1 for r in merged if r.get("source") == "gpt55"),
        "fallback_rows": sum(1 for r in merged if r.get("source") != "gpt55"),
        "api_error_rows": sum(1 for r in merged if r.get("api_error")),
        "workers": args.workers,
        "output": str(out_path),
        "uses_final_hidden_metadata": False,
        "uses_val41_label": False,
    }
    write_json(OUT_DIR / "gpt55_rule_distillation_summary.json", summary)
    report = CODEX / "reports" / f"V16_GPT55_RULE_DISTILLATION_{args.run_id}.md"
    report.write_text(
        "\n".join([
            f"# V16 GPT-5.5 Rule Distillation {args.run_id}",
            "",
            "- final_run: false",
            "- uses_final_hidden_metadata: false",
            "- uses_val41_label: false",
            f"- pool_available: {summary['pool_available']}",
            f"- samples_completed: {summary['samples_completed']}",
            f"- gpt55_rows: {summary['gpt55_rows']}",
            f"- fallback_rows: {summary['fallback_rows']}",
            f"- api_error_rows: {summary['api_error_rows']}",
            f"- workers: {summary['workers']}",
            f"- output: `{out_path}`",
        ]) + "\n",
        encoding="utf-8",
    )
    summary["report"] = str(report)
    write_json(OUT_DIR / "gpt55_rule_distillation_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
