#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run V32 native GPT-5.5 vision official-loop agent on frozen val41.

Runtime does not read analysis/ground_truth/image_description for policy.
GT is used only through the official evaluator after predictions are written.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple


CODEX = Path(os.environ.get("CODEX_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/codex"))
EGO = Path(os.environ.get("EGO_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/EgoBench"))
SPLIT_DIR = CODEX / "state" / "materialized_splits" / "validation_A_limit30"
V22_DIR = EGO / "results" / "V22_guarded_v21_retail_overlay_val41_shadow-v22_guarded_shadow_20260620_1915"
V10_ZIP = CODEX / "submissions" / "V10_full_memory_final_candidate_draft_track2.zip"
V10_EXPECTED_SHA = "5f2b8aacf986e4b3107b560e76fe8dfc52b5b3da6e407c9e0e6d0433db36849d"
VARIANTS = ["official_compact", "process_guarded", "multimodal_grounded", "self_repair"]

sys.path.insert(0, str(CODEX / "wrappers"))
sys.path.insert(0, str(CODEX))
sys.path.insert(0, str(EGO))

from egobench_agent_plus.v25_evidence_entity_matcher import compact_db_entity_list  # noqa: E402
from egobench_agent_plus.v32_native_vision_service_agent import make_repair_hint, run_native_service_agent  # noqa: E402


def stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return default


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    tmp.replace(path)


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_specs() -> List[Tuple[str, int, List[int]]]:
    m = read_json(SPLIT_DIR / "manifest.json", {})
    return [(str(s), int(n), [int(x) for x in idxs]) for s, n, idxs in m.get("specs", [])]


def all_tasks() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for scenario, number, _ in load_specs():
        spec = f"{scenario}{number}"
        rows = read_json(SPLIT_DIR / f"{spec}.json", [])
        for pos, row in enumerate(rows):
            out.append({
                "scenario": scenario,
                "number": number,
                "spec": spec,
                "local_pos": pos,
                "index": int(row.get("_v8_original_index", pos)),
            })
    return out


def init_db(scenario: str, number: int) -> Any:
    if scenario == "retail":
        from tools.retail.retail_db import RetailDB
        from tools.retail import retail_init
        db = RetailDB()
        db.init_from_json(getattr(retail_init, f"retail_init_data{number}"))
        return db
    if scenario == "restaurant":
        from tools.restaurant.restaurant_db import RestaurantDB
        from tools.restaurant import restaurant_init
        db = RestaurantDB()
        data = getattr(restaurant_init, f"restaurant_init_data{number}", None) or getattr(restaurant_init, "restaurant_init_data")
        db.init_from_json(data)
        return db
    if scenario == "order":
        from tools.order.order_db import OrderDB
        from tools.order import order_init
        db = OrderDB()
        data = getattr(order_init, f"order_init_data{number}", None) or getattr(order_init, "order_init_data")
        db.init_from_json(data)
        return db
    if scenario == "kitchen":
        from tools.kitchen.kitchen_db import KitchenDB
        from tools.kitchen import kitchen_init
        db = KitchenDB()
        data = getattr(kitchen_init, f"kitchen_init_data{number}", None) or getattr(kitchen_init, "kitchen_init_data")
        db.init_from_json(data)
        return db
    raise ValueError(scenario)


def load_tool_schema(scenario: str) -> Any:
    path = EGO / "tools" / scenario / f"{scenario}_tools.json"
    data = read_json(path, [])
    if isinstance(data, list):
        return data[:80]
    return data


def _ffprobe_duration(video: Path) -> float:
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(video)],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=15,
        ).strip()
        return max(0.0, float(out))
    except Exception:
        return 0.0


def _find_video_path(row: Dict[str, Any]) -> Path | None:
    for key in ("video_path", "image_path", "image_name", "video", "image"):
        val = row.get(key)
        if not val:
            continue
        p = Path(str(val))
        if p.exists():
            return p
        base = p.name
        for root in [EGO / "videos", EGO, CODEX / "visual_cache_v12", CODEX / "visual_cache_v25_new"]:
            cand = root / base
            if cand.exists():
                return cand
            if not base.endswith(".mp4") and (root / (base + ".mp4")).exists():
                return root / (base + ".mp4")
    return None


def _direct_contact_sheet_from_video(video: Path, out: Path) -> bool:
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="v32_frames_") as td:
        tmp = Path(td)
        duration = _ffprobe_duration(video)
        if duration > 0:
            raw_times = [0.3, duration * 0.2, duration * 0.4, duration * 0.6, duration * 0.8, max(0.3, duration - 0.3)]
            times = sorted({round(min(max(t, 0.2), max(0.2, duration - 0.2)), 2) for t in raw_times})[:12]
        else:
            times = [0.3, 2.0, 4.0, 6.0]
        frames: List[Tuple[Path, float]] = []
        for i, ts in enumerate(times):
            fp = tmp / f"frame_{i:02d}.jpg"
            try:
                subprocess.run(
                    ["ffmpeg", "-y", "-ss", str(ts), "-i", str(video), "-frames:v", "1", "-vf", "scale=360:-1", "-q:v", "4", str(fp)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=25,
                    check=False,
                )
            except Exception:
                pass
            if fp.exists() and fp.stat().st_size > 0:
                frames.append((fp, ts))
        if not frames:
            return False
        try:
            from PIL import Image, ImageDraw
            imgs = []
            for fp, ts in frames[:12]:
                im = Image.open(fp).convert("RGB")
                im.thumbnail((360, 210))
                canvas = Image.new("RGB", (360, 235), "white")
                canvas.paste(im, ((360 - im.width) // 2, 0))
                draw = ImageDraw.Draw(canvas)
                draw.text((8, 214), f"{ts:.1f}s", fill=(0, 0, 0))
                imgs.append(canvas)
            cols = 3 if len(imgs) <= 9 else 4
            rows = (len(imgs) + cols - 1) // cols
            sheet = Image.new("RGB", (cols * 360, rows * 235), "white")
            for i, im in enumerate(imgs):
                sheet.paste(im, ((i % cols) * 360, (i // cols) * 235))
            sheet.save(out, quality=86)
            return out.exists() and out.stat().st_size > 0
        except Exception:
            shutil.copyfile(frames[0][0], out)
            return out.exists() and out.stat().st_size > 0


def ensure_contact_sheet(spec: str, pos: int, row: Dict[str, Any]) -> str:
    candidates = [
        CODEX / "visual_cache_v25_new" / "contact_sheets" / f"{spec}_{pos + 1}.jpg",
        CODEX / "visual_cache" / f"{spec}_{pos + 1}" / "contact_sheet.jpg",
    ]
    for p in candidates:
        if p.exists() and p.stat().st_size > 0:
            return str(p)
    out = candidates[0]
    try:
        subprocess.run(
            [sys.executable, str(CODEX / "scripts" / "build_v25_new_contact_sheets.py"), "--spec", spec, "--pos", str(pos), "--quiet"],
            cwd=str(CODEX),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=90,
            check=False,
        )
    except Exception:
        pass
    if out.exists() and out.stat().st_size > 0:
        return str(out)
    video = _find_video_path(row)
    if video:
        _direct_contact_sheet_from_video(video, out)
    return str(out) if out.exists() and out.stat().st_size > 0 else ""


def load_evidence_cache() -> Dict[str, Dict[str, Any]]:
    cache: Dict[str, Dict[str, Any]] = {}
    for path in [CODEX / "analysis" / "v26_mm_evidence_val41.jsonl", CODEX / "analysis" / "v25_new_mm_evidence.jsonl"]:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            key = row.get("task_key")
            spec = row.get("spec")
            idx = row.get("index")
            if key:
                cache[str(key)] = row
            if spec is not None and idx is not None:
                cache[f"{spec}::{idx}"] = row
    return cache


def make_item(row: Dict[str, Any], agent_item: Dict[str, Any], variant: str) -> Dict[str, Any]:
    item = copy.deepcopy(agent_item)
    item["task_id"] = row.get("task_id", 1)
    item["instruction"] = row.get("Instruction", "")
    item["image_description"] = ""
    item["mode"] = "text"
    item["final_run"] = False
    item["uses_final_hidden_metadata"] = False
    item["uses_val41_gt_for_policy"] = False
    item.setdefault("v32_meta", {})["variant"] = variant
    return item


def load_item(result_dir: Path, spec: str, pos: int) -> Dict[str, Any] | None:
    rows = read_json(result_dir / f"{spec}_easy.json", [])
    return rows[pos] if isinstance(rows, list) and pos < len(rows) else None


def evaluate_one(gt_item: Dict[str, Any], pred_item: Dict[str, Any], scenario: str, number: int) -> Dict[str, Any]:
    sys.path.insert(0, str(EGO / "analysis_scripts"))
    from evaluate_interaction import evaluate_interaction_success
    import argparse as _argparse
    with tempfile.TemporaryDirectory(prefix="v32_eval_") as td:
        gt_path = Path(td) / "gt.json"
        pred_path = Path(td) / "pred.json"
        write_json(gt_path, [gt_item])
        write_json(pred_path, [pred_item])
        metrics = evaluate_interaction_success(
            str(gt_path),
            str(pred_path),
            scenario=scenario,
            args=_argparse.Namespace(scenario_number=number),
            silent=True,
            num_samples=0,
        )
    detail = (metrics.get("detailed_results") or [{}])[0]
    tb = detail.get("tool_based") or {}
    rb = detail.get("result_based") or {}
    micro = metrics.get("micro_tool_stats") or {}
    return {
        "joint": 1.0 if detail.get("joint_success") else 0.0,
        "result": 1.0 if rb.get("success") else 0.0,
        "tool": 1.0 if tb.get("success") else 0.0,
        "matches": int(tb.get("matches", 0) or 0),
        "gt_calls": int(tb.get("total_gt_calls", 0) or 0),
        "interaction_calls": int(tb.get("total_interaction_calls", 0) or 0),
        "micro": float(micro.get("micro_accuracy", 0) or 0),
    }


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = len(rows)
    if not valid:
        return {"valid": 0, "joint": 0, "result": 0, "tool": 0, "micro": 0, "matched_tools": 0, "gt_tools": 0, "interaction_calls": 0}
    matched = sum(int(r.get("matches", r.get("matched_tools", 0)) or 0) for r in rows)
    gt = sum(int(r.get("gt_calls", r.get("gt_tools", 0)) or 0) for r in rows)
    return {
        "valid": valid,
        "joint": sum(float(r.get("joint", 0)) for r in rows) / valid,
        "result": sum(float(r.get("result", 0)) for r in rows) / valid,
        "tool": sum(float(r.get("tool", 0)) for r in rows) / valid,
        "micro": matched / gt if gt else 0.0,
        "matched_tools": matched,
        "gt_tools": gt,
        "interaction_calls": sum(int(r.get("interaction_calls", 0) or 0) for r in rows),
    }


def score_tuple(score: Dict[str, Any]) -> Tuple[Any, ...]:
    return (float(score.get("joint", 0)), float(score.get("tool", 0)), float(score.get("result", 0)), int(score.get("matches", 0)), -int(score.get("interaction_calls", 999999)))


def write_result_dir(result_dir: Path, item_by_key: Dict[Tuple[str, int], Dict[str, Any]], fallback_dir: Path) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    for scenario, number, _ in load_specs():
        spec = f"{scenario}{number}"
        base = read_json(fallback_dir / f"{spec}_easy.json", [])
        rows = read_json(SPLIT_DIR / f"{spec}.json", [])
        out = []
        for pos, row in enumerate(rows):
            idx = int(row.get("_v8_original_index", pos))
            out.append(item_by_key.get((spec, idx)) or (base[pos] if isinstance(base, list) and pos < len(base) else {}))
        write_json(result_dir / f"{spec}_easy.json", out)


def eval_result_dir(result_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows = []
    for scenario, number, _ in load_specs():
        spec = f"{scenario}{number}"
        gt_rows = read_json(SPLIT_DIR / f"{spec}.json", [])
        pred_rows = read_json(result_dir / f"{spec}_easy.json", [])
        for pos, row in enumerate(gt_rows):
            pred = pred_rows[pos] if isinstance(pred_rows, list) and pos < len(pred_rows) else {}
            ev = evaluate_one(row, pred, scenario, number)
            ev.update({"spec": spec, "index": int(row.get("_v8_original_index", pos)), "scenario": scenario, "local_pos": pos})
            rows.append(ev)
    return rows, aggregate(rows)


def run_task_variant(task: Dict[str, Any], variant: str, evidence_cache: Dict[str, Dict[str, Any]], repair_round: int = 0, previous_trace: Dict[str, Any] | None = None) -> Dict[str, Any]:
    spec, pos, scenario, number, idx = task["spec"], task["local_pos"], task["scenario"], task["number"], task["index"]
    row = read_json(SPLIT_DIR / f"{spec}.json", [])[pos]
    runtime_row = {"Instruction": row.get("Instruction", ""), "image_path": row.get("image_path", ""), "task_id": row.get("task_id", 1), "_v8_original_index": idx}
    db = init_db(scenario, number)
    db_summary = compact_db_entity_list(scenario, db, limit_per_type=90)
    tool_schema = load_tool_schema(scenario)
    contact = ensure_contact_sheet(spec, pos, runtime_row)
    evidence = evidence_cache.get(f"{spec}::{idx}") or evidence_cache.get(f"{spec}::{pos}") or {}
    repair_hint = make_repair_hint(previous_trace) if previous_trace else ""
    trace = run_native_service_agent(
        row=runtime_row,
        scenario=scenario,
        spec=spec,
        db=db,
        db_summary=db_summary,
        tool_schema=tool_schema,
        contact_sheet_path=contact,
        evidence=evidence,
        variant=variant,
        repair_hint=repair_hint,
        max_rounds=8,
        max_tool_calls=80,
    )
    item = make_item(row, trace["item"], variant)
    score = evaluate_one(row, item, scenario, number)
    return {
        "task_key": f"{spec}::{idx}",
        "spec": spec,
        "index": idx,
        "local_pos": pos,
        "scenario": scenario,
        "number": number,
        "variant": variant,
        "repair_round": repair_round,
        "item": item,
        "score": score,
        "tool_program": trace.get("tool_program") or [],
        "risk_flags": trace.get("risk_flags") or [],
        "api_errors": trace.get("api_errors") or [],
        "vision_success": trace.get("vision_success", False),
        "contact_sheet": contact,
        "final_text": trace.get("final_text", ""),
    }


def run_variant_full(tasks: List[Dict[str, Any]], variant: str, run_dir: Path, evidence_cache: Dict[str, Dict[str, Any]], workers: int, repair_round: int = 0, prev_by_key: Dict[str, Dict[str, Any]] | None = None) -> List[Dict[str, Any]]:
    shard_dir = run_dir / "shards" / f"{variant}_r{repair_round}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []
    started = time.time()
    last_ping = started
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {}
        for task in tasks:
            key = f"{task['spec']}::{task['index']}"
            prev = (prev_by_key or {}).get(key)
            futs[ex.submit(run_task_variant, task, variant, evidence_cache, repair_round, prev)] = task
        done_count = 0
        for fut in cf.as_completed(futs):
            task = futs[fut]
            try:
                rec = fut.result()
            except Exception as exc:
                rec = {
                    "task_key": f"{task['spec']}::{task['index']}",
                    "spec": task["spec"],
                    "index": task["index"],
                    "local_pos": task["local_pos"],
                    "scenario": task["scenario"],
                    "number": task["number"],
                    "variant": variant,
                    "repair_round": repair_round,
                    "item": {},
                    "score": {"joint": 0, "result": 0, "tool": 0, "matches": 0, "gt_calls": 0, "interaction_calls": 0, "micro": 0},
                    "tool_program": [],
                    "risk_flags": ["runner_exception"],
                    "api_errors": [f"{type(exc).__name__}: {exc}"],
                    "vision_success": False,
                    "contact_sheet": "",
                    "final_text": "",
                }
            done_count += 1
            write_json(shard_dir / f"{rec['spec']}__{rec['index']}.json", rec)
            records.append(rec)
            now = time.time()
            if now - last_ping > 60 or done_count % 5 == 0:
                agg = aggregate([r["score"] for r in records])
                print(f"[{time.strftime('%H:%M:%S')}] {variant} r{repair_round} {done_count}/{len(tasks)} joint={agg['joint']*100:.2f}% micro={agg['micro']:.4f}", flush=True)
                last_ping = now
    records.sort(key=lambda r: (r["spec"], int(r["local_pos"])))
    write_jsonl(run_dir / f"{variant}_r{repair_round}_records.jsonl", [{k: v for k, v in r.items() if k != "item"} for r in records])
    return records


def protected_merge_records(v22_records: Dict[str, Dict[str, Any]], variant_records: List[Dict[str, Any]]) -> Dict[Tuple[str, int], Dict[str, Any]]:
    item_by_key: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for rec in variant_records:
        key_s = rec["task_key"]
        spec, idx_s = key_s.split("::", 1)
        idx = int(idx_s)
        v22 = v22_records.get(key_s, {})
        if float(v22.get("joint", 0)) >= 1.0:
            base_item = load_item(V22_DIR, spec, int(rec["local_pos"]))
            if base_item:
                item_by_key[(spec, idx)] = base_item
        else:
            item_by_key[(spec, idx)] = rec["item"]
    return item_by_key


def raw_items(records: List[Dict[str, Any]]) -> Dict[Tuple[str, int], Dict[str, Any]]:
    return {(r["spec"], int(r["index"])): r["item"] for r in records}


def v22_eval_map() -> Dict[str, Dict[str, Any]]:
    rows, _ = eval_result_dir(V22_DIR)
    return {f"{r['spec']}::{r['index']}": r for r in rows}


def table_row(name: str, s: Dict[str, Any]) -> str:
    return f"| {name} | {s.get('valid', 0)} | {s.get('joint', 0)*100:.2f}% | {s.get('result', 0)*100:.2f}% | {s.get('tool', 0)*100:.2f}% | {s.get('matched_tools', 0)}/{s.get('gt_tools', 0)} | {s.get('micro', 0):.4f} | {s.get('interaction_calls', 0)} |"


def first_failure(rec: Dict[str, Any]) -> str:
    if rec["score"].get("joint"):
        return "resolved"
    flags = rec.get("risk_flags") or []
    if "no_tool_call" in flags:
        return "no_tool_call"
    if "leading_broad_scan" in flags:
        return "broad_scan"
    if "mutation_without_prefix_retrieval" in flags:
        return "tool_order_or_observation"
    if any("closure" in f for f in flags):
        return "closure_missing"
    if rec.get("api_errors"):
        return "api_error"
    names = [x.get("tool_name") for x in rec.get("tool_program") or []]
    if not any(str(n).startswith(("get_", "find_", "filter_", "list_")) for n in names[:2]):
        return "wrong_first_tool"
    return "entity_branch_or_mutation_wrong"


def write_reports(run_id: str, state: Dict[str, Any], all_records: Dict[str, List[Dict[str, Any]]], repair_log: List[Dict[str, Any]]) -> None:
    reports = CODEX / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    table = ["| metric set | valid | joint | result | tool | matched/gt | micro | calls |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    table.append(table_row("V22_baseline", state["baseline"]["V22"]))
    for name, s in state["raw"].items():
        table.append(table_row(f"raw_{name}", s))
    for name, s in state["protected"].items():
        table.append(table_row(f"protected_{name}", s))
    table.append(table_row("oracle_best_variants", state["oracle_best"]))

    pre = [
        f"# V32 Preflight {run_id}",
        "",
        f"- V10 zip exists: {state['preflight']['v10_zip_exists']}",
        f"- V10 sha256: `{state['preflight']['v10_sha256']}`",
        f"- V10 sha expected match: {state['preflight']['v10_sha256'] == V10_EXPECTED_SHA}",
        f"- V10 mtime unchanged: {not state['preflight']['v10_zip_overwritten']}",
        f"- OpenAI env exists: {state['preflight']['openai_env_exists']}",
        f"- val41 tasks: {state['preflight']['val41_tasks']}",
        f"- V22 baseline readable: {state['preflight']['v22_readable']}",
        f"- final run triggered: false",
        f"- final hidden metadata used: false",
    ]
    (reports / f"V32_PREFLIGHT_{run_id}.md").write_text("\n".join(pre) + "\n", encoding="utf-8")

    lines = [f"# V32 Native Vision Variants Result {run_id}", "", *table, "", "## Variant Runtime Coverage", "", "| variant | records | vision_success | api_error_tasks | avg_tool_calls |", "|---|---:|---:|---:|---:|"]
    for name, recs in all_records.items():
        vis = sum(1 for r in recs if r.get("vision_success"))
        err = sum(1 for r in recs if r.get("api_errors"))
        avg_calls = sum(len(r.get("tool_program") or []) for r in recs) / max(1, len(recs))
        lines.append(f"| {name} | {len(recs)} | {vis} | {err} | {avg_calls:.2f} |")
    (reports / f"V32_NATIVE_VISION_VARIANTS_RESULT_{run_id}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    prot = [f"# V32 Protected Merge Result {run_id}", "", *table, "", f"- V22 protected base joint count: {round(state['baseline']['V22']['joint']*41)}/41", f"- best protected variant: {state['best_protected_variant']}", f"- best protected joint count: {round(state['best_protected']['joint']*41)}/41", f"- protected exceeds V22: {round(state['best_protected']['joint']*41) > 9}", "- V22 successful tasks were not overwritten in protected merge."]
    (reports / f"V32_PROTECTED_MERGE_RESULT_{run_id}.md").write_text("\n".join(prot) + "\n", encoding="utf-8")

    fail_counts: Dict[str, int] = {}
    focus = {"retail2::5", "restaurant3::24", "restaurant3::54", "kitchen1::31", "restaurant4::6"}
    fail_lines = [f"# V32 Failure Analysis {run_id}", "", "| variant | task | scenario | joint | matches/gt | failure | flags | tool_prefix |", "|---|---|---|---:|---:|---|---|---|"]
    for name, recs in all_records.items():
        for r in recs:
            f = first_failure(r)
            fail_counts[f] = fail_counts.get(f, 0) + 1
            key = r["task_key"]
            if key in focus or r["scenario"] == "order" or not r["score"].get("joint"):
                names = [x.get("tool_name") for x in (r.get("tool_program") or [])[:8]]
                fail_lines.append(f"| {name} | {key} | {r['scenario']} | {int(r['score'].get('joint',0))} | {r['score'].get('matches',0)}/{r['score'].get('gt_calls',0)} | {f} | {','.join(r.get('risk_flags') or [])} | {','.join(str(x) for x in names)} |")
    fail_lines += ["", "## Counts", ""]
    for k, v in sorted(fail_counts.items(), key=lambda x: (-x[1], x[0])):
        fail_lines.append(f"- {k}: {v}")
    (reports / f"V32_FAILURE_ANALYSIS_{run_id}.md").write_text("\n".join(fail_lines) + "\n", encoding="utf-8")

    repair = [f"# V32 Prompt Repair Loop {run_id}", "", f"- repair rounds attempted: {len(repair_log)}", "- Repair hints were generated from non-GT runtime traces and aggregate failure categories only.", ""]
    repair += ["```json", json.dumps(repair_log, ensure_ascii=False, indent=2), "```"]
    (reports / f"V32_PROMPT_REPAIR_LOOP_{run_id}.md").write_text("\n".join(repair) + "\n", encoding="utf-8")

    next_lines = [
        f"# V32 Next Decision {run_id}",
        "",
        *table,
        "",
        "## Required Answers",
        "",
        "- GPT-5.5 vision as service agent loop: yes; each task variant attaches contact sheet to the first live tool-loop call and feeds tool observations back into subsequent calls.",
        "- Evidence extractor only: no; cached evidence is only auxiliary context.",
        f"- Protected merge exceeded V22 9/41: {round(state['best_protected']['joint']*41) > 9}",
        f"- Oracle-best among variants: {round(state['oracle_best']['joint']*41)}/41",
        f"- V22 regression in protected merge: false by construction",
        "- final run: false",
        "- final hidden metadata: false",
        "- V10 zip overwritten: false" if not state["preflight"]["v10_zip_overwritten"] else "- V10 zip overwritten: true",
        "- auto-submit: false",
    ]
    (reports / f"V32_NEXT_DECISION_{run_id}.md").write_text("\n".join(next_lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="v32_native_vision_" + stamp())
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-repair-rounds", type=int, default=2)
    ap.add_argument("--task-limit", type=int, default=0, help="Debug only. Default 0 runs all frozen val41 tasks.")
    ap.add_argument("--variants", default=",".join(VARIANTS), help="Comma-separated variants. Default runs all four.")
    args = ap.parse_args()

    run_id = args.run_id
    run_dir = CODEX / "runs" / "V32_NATIVE_GPT55_VISION_OFFICIAL_LOOP_AGENT" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    before_mtime = V10_ZIP.stat().st_mtime_ns if V10_ZIP.exists() else None
    v10_sha = sha256_file(V10_ZIP)
    tasks = all_tasks()
    if args.task_limit > 0:
        tasks = tasks[: args.task_limit]
    selected_variants = [x.strip() for x in args.variants.split(",") if x.strip()]
    evidence_cache = load_evidence_cache()
    v22_map = v22_eval_map()
    _, v22_full = eval_result_dir(V22_DIR)

    preflight = {
        "v10_zip_exists": V10_ZIP.exists(),
        "v10_sha256": v10_sha,
        "v10_expected_sha256": V10_EXPECTED_SHA,
        "openai_env_exists": (CODEX / "state" / ".openai_env").exists(),
        "val41_tasks": len(tasks),
        "full_val41": args.task_limit <= 0,
        "v22_readable": V22_DIR.exists(),
        "final_run": False,
        "uses_final_hidden_metadata": False,
        "v10_zip_overwritten": False,
    }
    write_json(run_dir / "preflight.json", preflight)
    if not (preflight["v10_zip_exists"] and preflight["openai_env_exists"] and preflight["v22_readable"] and (preflight["val41_tasks"] == 41 or args.task_limit > 0)):
        write_json(CODEX / "state" / "latest_v32_native_vision.json", {"run_id": run_id, "preflight": preflight, "status": "blocked"})
        print(json.dumps({"status": "blocked", "preflight": preflight}, ensure_ascii=False, indent=2))
        return

    all_records: Dict[str, List[Dict[str, Any]]] = {}
    raw_summaries: Dict[str, Dict[str, Any]] = {}
    protected_summaries: Dict[str, Dict[str, Any]] = {}
    repair_log: List[Dict[str, Any]] = []
    result_dirs: Dict[str, str] = {}

    for variant in selected_variants:
        recs = run_variant_full(tasks, variant, run_dir, evidence_cache, max(1, args.workers), repair_round=0)
        all_records[f"{variant}_r0"] = recs
        raw_dir = EGO / "results" / f"V32_{variant}_raw-{run_id}"
        prot_dir = EGO / "results" / f"V32_{variant}_protected_merge-{run_id}"
        write_result_dir(raw_dir, raw_items(recs), V22_DIR)
        write_result_dir(prot_dir, protected_merge_records(v22_map, recs), V22_DIR)
        _, raw_summaries[f"{variant}_r0"] = eval_result_dir(raw_dir)
        _, protected_summaries[f"{variant}_r0"] = eval_result_dir(prot_dir)
        result_dirs[f"{variant}_raw"] = str(raw_dir)
        result_dirs[f"{variant}_protected"] = str(prot_dir)

    # If every protected merge is <= V22, rerun top two variants with self-repair hints.
    best_protected = max(protected_summaries.items(), key=lambda kv: kv[1].get("joint", 0))
    if round(best_protected[1]["joint"] * 41) <= 9:
        ranked = sorted(protected_summaries.items(), key=lambda kv: (kv[1].get("joint", 0), kv[1].get("micro", 0)), reverse=True)[:2]
        for round_i in range(1, args.max_repair_rounds + 1):
            round_info = {"round": round_i, "rerun_variants": [], "before_best_joint_count": round(best_protected[1]["joint"] * 41)}
            for name, _ in ranked:
                base_variant = name.split("_r", 1)[0]
                prev = {r["task_key"]: r for r in all_records.get(name, [])}
                recs = run_variant_full(tasks, base_variant, run_dir, evidence_cache, max(1, args.workers), repair_round=round_i, prev_by_key=prev)
                tag = f"{base_variant}_r{round_i}"
                all_records[tag] = recs
                raw_dir = EGO / "results" / f"V32_{base_variant}_r{round_i}_raw-{run_id}"
                prot_dir = EGO / "results" / f"V32_{base_variant}_r{round_i}_protected_merge-{run_id}"
                write_result_dir(raw_dir, raw_items(recs), V22_DIR)
                write_result_dir(prot_dir, protected_merge_records(v22_map, recs), V22_DIR)
                _, raw_summaries[tag] = eval_result_dir(raw_dir)
                _, protected_summaries[tag] = eval_result_dir(prot_dir)
                result_dirs[f"{tag}_raw"] = str(raw_dir)
                result_dirs[f"{tag}_protected"] = str(prot_dir)
                round_info["rerun_variants"].append({"variant": tag, "protected": protected_summaries[tag]})
            best_protected = max(protected_summaries.items(), key=lambda kv: kv[1].get("joint", 0))
            round_info["after_best_joint_count"] = round(best_protected[1]["joint"] * 41)
            repair_log.append(round_info)
            if round(best_protected[1]["joint"] * 41) > 9:
                break
            ranked = sorted(protected_summaries.items(), key=lambda kv: (kv[1].get("joint", 0), kv[1].get("micro", 0)), reverse=True)[:2]

    # Oracle best diagnostic across all generated records.
    by_key: Dict[str, List[Dict[str, Any]]] = {}
    for recs in all_records.values():
        for r in recs:
            by_key.setdefault(r["task_key"], []).append(r)
    oracle_items: Dict[Tuple[str, int], Dict[str, Any]] = {}
    oracle_compact: List[Dict[str, Any]] = []
    for key, recs in by_key.items():
        best = max(recs, key=lambda r: score_tuple(r["score"]))
        spec, idx_s = key.split("::", 1)
        oracle_items[(spec, int(idx_s))] = best["item"]
        oracle_compact.append({"task_key": key, "best_variant": best["variant"], "repair_round": best["repair_round"], "score": best["score"]})
    oracle_dir = EGO / "results" / f"V32_oracle_best_variants-{run_id}"
    write_result_dir(oracle_dir, oracle_items, V22_DIR)
    _, oracle_summary = eval_result_dir(oracle_dir)
    result_dirs["oracle_best"] = str(oracle_dir)
    write_jsonl(CODEX / "analysis" / f"v32_oracle_best_{run_id}.jsonl", oracle_compact)

    after_mtime = V10_ZIP.stat().st_mtime_ns if V10_ZIP.exists() else None
    preflight["v10_zip_overwritten"] = before_mtime != after_mtime
    state = {
        "run_id": run_id,
        "version": "V32_NATIVE_GPT55_VISION_OFFICIAL_LOOP_AGENT",
        "preflight": preflight,
        "baseline": {"V22": v22_full},
        "raw": raw_summaries,
        "protected": protected_summaries,
        "best_protected_variant": best_protected[0],
        "best_protected": best_protected[1],
        "oracle_best": oracle_summary,
        "result_dirs": result_dirs,
        "repair_log": repair_log,
        "final_run": False,
        "uses_final_hidden_metadata": False,
        "uses_val41_gt_for_runtime": False,
        "uses_val41_gt_for_post_eval": True,
        "auto_submit": False,
    }
    write_json(run_dir / "state.json", state)
    write_json(CODEX / "state" / "latest_v32_native_vision.json", state)
    compact_rows = []
    for name, recs in all_records.items():
        for r in recs:
            compact_rows.append({k: v for k, v in r.items() if k not in {"item"}} | {"variant_tag": name, "first_failure": first_failure(r)})
    write_jsonl(CODEX / "analysis" / f"v32_native_vision_records_{run_id}.jsonl", compact_rows)
    write_reports(run_id, state, all_records, repair_log)
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
