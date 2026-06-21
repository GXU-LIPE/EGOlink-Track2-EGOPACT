#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V17 GPT-5.5 distillation with hard fail-fast validation."""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import ssl
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Tuple


CODEX = Path("/home/data-gxu/acm/egolink2026-main/code/track2/codex")
GT16 = CODEX / "gt_distill_v16"
GT17 = CODEX / "gt_distill_v17"
REQUIRED_KEYS = {
    "sample_id",
    "scenario",
    "task_type",
    "intent_type",
    "minimal_tool_skeleton",
    "required_slots",
    "slot_resolution_rules",
    "branch_compiler_rules",
    "closure_rules",
    "anti_patterns",
    "generalizable_rule",
}


def load_env_file(path: Path) -> Dict[str, str]:
    env = {}
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line.startswith("export ") and "=" in line:
            k, v = line[len("export "):].split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl_atomic(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    tmp.replace(path)


def compact_sample(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "sample_id": row.get("pool_id"),
        "scenario": row.get("scenario"),
        "spec": row.get("spec"),
        "task_type": row.get("task_type"),
        "instruction": (row.get("instruction") or "")[:1200],
        "analysis": (row.get("analysis") or "")[:1200],
        "tool_names": row.get("tool_names"),
        "tool_family_sequence": row.get("tool_family_sequence"),
        "entity_slots": row.get("entity_slots"),
        "tool_calls": row.get("tool_calls"),
    }


def prompt_for(row: Dict[str, Any]) -> str:
    schema = {
        "sample_id": "...",
        "scenario": "...",
        "task_type": "...",
        "intent_type": "query_only|visual_query|branch_then_mutation|ranking_filtering|cart_order_mutation|aggregate_required",
        "minimal_tool_skeleton": [{"stage": "pin|retrieve|canonicalize|branch_check|mutation|aggregate|answer", "allowed_tools": ["..."], "required": True}],
        "required_slots": {"user_id": "", "restaurant_name": "", "entity_type": "", "canonical_entity": "", "category": "", "branch_condition": "", "mutation_target": "", "aggregate_type": ""},
        "slot_resolution_rules": [{"phrase_pattern": "...", "slot": "...", "resolver": "tool|ledger|visual|db_lookup", "canonicalization": "..."}],
        "branch_compiler_rules": [{"condition": "...", "check_tools": ["..."], "if_true_tools": ["..."], "if_false_tools": ["..."]}],
        "closure_rules": [{"missing": "...", "repair_tool": "...", "when": "..."}],
        "anti_patterns": ["unconstrained broad scan", "mutation before canonicalization", "missing final aggregate", "wrong active restaurant", "dish/set_meal confusion"],
        "generalizable_rule": "...",
    }
    return (
        "Distill this EgoBench Track2 non-final, non-val41 GT trajectory into executable compiler rules. "
        "Do not use final hidden metadata. Do not output markdown. Return strict JSON matching this schema:\n"
        + json.dumps(schema, ensure_ascii=False)
        + "\n\nSample:\n"
        + json.dumps(compact_sample(row), ensure_ascii=False)
    )


def parse_rule(text: str, row: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", str(text).strip(), flags=re.I | re.S)
    obj = json.loads(cleaned)
    missing = sorted(REQUIRED_KEYS - set(obj))
    if missing:
        raise ValueError("missing keys: " + ",".join(missing))
    if not isinstance(obj.get("minimal_tool_skeleton"), list):
        raise ValueError("minimal_tool_skeleton not list")
    obj["sample_id"] = obj.get("sample_id") or row.get("pool_id")
    obj["scenario"] = obj.get("scenario") or row.get("scenario")
    obj["no_final_metadata"] = True
    obj["uses_val41_label"] = False
    return True, obj


def call_one(row: Dict[str, Any], env: Dict[str, str], timeout: int, retries: int) -> Dict[str, Any]:
    key = env.get("OPENAI_API_KEY") or env.get("SERVICE_MODEL_API_KEY")
    base = env.get("TRACK2_OPENAI_BASE_URL") or env.get("SERVICE_MODEL_API_BASE") or "https://ai-pixel.online/v1"
    model = env.get("TRACK2_OPENAI_MODEL") or env.get("SERVICE_MODEL_NAME") or "gpt-5.5"
    if not key:
        return {"sample_id": row.get("pool_id"), "scenario": row.get("scenario"), "valid": False, "error": "missing_api_key"}
    url = base.rstrip("/")
    if not url.endswith("/chat/completions"):
        url += "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return strict JSON only. No markdown."},
            {"role": "user", "content": prompt_for(row)},
        ],
        "temperature": 0,
        "max_tokens": 1100,
    }
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        urllib.request.HTTPSHandler(context=ssl.create_default_context()),
    )
    last_error = ""
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
                method="POST",
            )
            with opener.open(req, timeout=timeout) as resp:
                body = json.loads(resp.read().decode("utf-8", errors="replace"))
            text = body.get("choices", [{}])[0].get("message", {}).get("content", "")
            _ok, obj = parse_rule(text, row)
            obj["valid"] = True
            obj["api_model"] = model
            obj["attempts"] = attempt + 1
            return obj
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {str(exc)[:220]}"
            time.sleep(min(2 + attempt * 2, 6))
    return {"sample_id": row.get("pool_id"), "scenario": row.get("scenario"), "valid": False, "error": last_error}


def worker(shard_id: int, rows: List[Dict[str, Any]], env: Dict[str, str], timeout: int, retries: int, run_id: str) -> Dict[str, Any]:
    shard_dir = GT17 / "gpt55_shards" / run_id
    log_path = shard_dir / f"shard_{shard_id:02d}.log"
    out_path = shard_dir / f"shard_{shard_id:02d}.jsonl"
    heartbeat = shard_dir / f"shard_{shard_id:02d}.heartbeat.json"
    done = []
    start = time.time()
    shard_dir.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(rows, start=1):
        rec = call_one(row, env, timeout, retries)
        done.append(rec)
        heartbeat.write_text(json.dumps({"shard": shard_id, "done": i, "total": len(rows), "seconds": round(time.time() - start, 1)}, ensure_ascii=False), encoding="utf-8")
        with log_path.open("a", encoding="utf-8") as log:
            log.write(json.dumps({"i": i, "sample_id": row.get("pool_id"), "valid": rec.get("valid"), "error": rec.get("error", "")}, ensure_ascii=False) + "\n")
    append_jsonl_atomic(out_path, done)
    return {"shard": shard_id, "rows": len(done), "valid": sum(1 for r in done if r.get("valid")), "out": str(out_path)}


def write_report(run_id: str, summary: Dict[str, Any]) -> Path:
    report = CODEX / "reports" / f"V17_GPT55_DISTILLATION_CHECK_{run_id}.md"
    lines = [
        f"# V17 GPT-5.5 Distillation Check {run_id}",
        "",
        f"- status: {summary['status']}",
        "- final_run: false",
        "- uses_final_hidden_metadata: false",
        "- uses_val41_label: false",
        "- deterministic_fallback_allowed: false",
        "",
        "## Metrics",
        "",
        f"- pool_rows: {summary.get('pool_rows')}",
        f"- attempted_rows: {summary.get('attempted_rows')}",
        f"- gpt55_rows: {summary.get('gpt55_rows')}",
        f"- invalid_rows: {summary.get('invalid_rows')}",
        f"- parse_valid_rate: {summary.get('parse_valid_rate'):.4f}",
        f"- scenario_coverage: {summary.get('scenario_coverage')}",
        f"- workers: {summary.get('workers')}",
    ]
    if summary.get("fail_reasons"):
        lines += ["", "## Fail Reasons", ""]
        lines.extend(f"- {x}" for x in summary["fail_reasons"])
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default=f"v17_gpt55_distill_{time.strftime('%Y%m%d_%H%M%S')}")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--timeout", type=int, default=45)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="0 means all rows; hard gate still requires >=600 gpt rows")
    args = ap.parse_args()

    pool = read_jsonl(GT16 / "gt100_pool.jsonl")
    if args.limit:
        pool = pool[: args.limit]
    env = os.environ.copy()
    env.update(load_env_file(CODEX / "state" / ".openai_env"))
    shards = [[] for _ in range(max(1, args.workers))]
    for i, row in enumerate(pool):
        shards[i % len(shards)].append(row)
    start = time.time()
    results = []
    with cf.ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = [ex.submit(worker, i, shard, env, args.timeout, args.retries, args.run_id) for i, shard in enumerate(shards) if shard]
        for fut in cf.as_completed(futs):
            item = fut.result()
            results.append(item)
            print(json.dumps(item, ensure_ascii=False), flush=True)
    merged = []
    shard_dir = GT17 / "gpt55_shards" / args.run_id
    for path in sorted(shard_dir.glob("shard_*.jsonl")):
        merged.extend(read_jsonl(path))
    out_path = GT17 / "gpt55_distilled_rules.jsonl"
    append_jsonl_atomic(out_path, merged)
    valid = [r for r in merged if r.get("valid")]
    scenarios = sorted({r.get("scenario") for r in valid if r.get("scenario")})
    parse_valid_rate = len(valid) / len(merged) if merged else 0.0
    fail = []
    if len(valid) < 600:
        fail.append(f"gpt55_rows {len(valid)} < 600")
    if parse_valid_rate < 0.95:
        fail.append(f"parse_valid_rate {parse_valid_rate:.4f} < 0.95")
    missing = sorted(set(["order", "restaurant", "retail", "kitchen"]) - set(scenarios))
    if missing:
        fail.append("missing scenario coverage: " + ",".join(missing))
    summary = {
        "run_id": args.run_id,
        "status": "PASS" if not fail else "FAIL",
        "pool_rows": len(read_jsonl(GT16 / "gt100_pool.jsonl")),
        "attempted_rows": len(merged),
        "gpt55_rows": len(valid),
        "invalid_rows": len(merged) - len(valid),
        "parse_valid_rate": parse_valid_rate,
        "scenario_coverage": scenarios,
        "workers": args.workers,
        "seconds": round(time.time() - start, 1),
        "fail_reasons": fail,
        "output": str(out_path),
        "uses_final_hidden_metadata": False,
        "uses_val41_label": False,
        "deterministic_fallback_allowed": False,
    }
    report = write_report(args.run_id, summary)
    summary["report"] = str(report)
    write_json(GT17 / "gpt55_distillation_manifest.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if fail:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
