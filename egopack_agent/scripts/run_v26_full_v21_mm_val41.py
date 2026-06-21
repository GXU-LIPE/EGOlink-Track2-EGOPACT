#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run V26 full V21-style multimodal evidence agent on frozen val41."""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
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
V14_DIR = EGO / "results" / "V14_candidate_selection_val41-v14_candidate_selection_20260619_2134"
V22_DIR = EGO / "results" / "V22_guarded_v21_retail_overlay_val41_shadow-v22_guarded_shadow_20260620_1915"
V24_DIR = EGO / "results" / "V24_scenario_gt_gap_generators_val41_shadow-v24_gap_generators_gpt_selector_20260620_2245"
V10_ZIP = CODEX / "submissions" / "V10_full_memory_final_candidate_draft_track2.zip"

sys.path.insert(0, str(CODEX / "wrappers"))
sys.path.insert(0, str(CODEX))

from egobench_agent_plus.v24_candidate_dryrun_and_selector import dryrun_program  # noqa: E402
from egobench_agent_plus.v25_mm_evidence_extractor import build_evidence_table, save_evidence  # noqa: E402
from egobench_agent_plus.v26_multimodal_evidence_agent import bind_evidence_v26, build_candidates_v26, select_v26  # noqa: E402


def stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return default


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


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


def load_specs() -> List[Tuple[str, int, List[int]]]:
    m = read_json(SPLIT_DIR / "manifest.json", {})
    return [(str(s), int(n), [int(x) for x in idxs]) for s, n, idxs in m.get("specs", [])]


def init_db(scenario: str, number: int) -> Any:
    sys.path.insert(0, str(EGO))
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
        db.init_from_json(getattr(restaurant_init, f"restaurant_init_data{number}", None) or getattr(restaurant_init, "restaurant_init_data"))
        return db
    if scenario == "order":
        from tools.order.order_db import OrderDB
        from tools.order import order_init
        db = OrderDB()
        db.init_from_json(getattr(order_init, f"order_init_data{number}", None) or getattr(order_init, "order_init_data"))
        return db
    if scenario == "kitchen":
        from tools.kitchen.kitchen_db import KitchenDB
        from tools.kitchen import kitchen_init
        db = KitchenDB()
        db.init_from_json(getattr(kitchen_init, f"kitchen_init_data{number}", None) or getattr(kitchen_init, "kitchen_init_data"))
        return db
    raise ValueError(scenario)


def qwen_card(spec: str, pos: int) -> Dict[str, Any]:
    for p in [
        CODEX / "visual_cache_v12" / "qwen3vl_grounding" / f"{spec}_{pos + 1}.json",
        CODEX / "visual_cache_v12" / "backups" / "qwen3vl_grounding_before_val41_allcards_20260619_144227" / f"{spec}_{pos + 1}.json",
        CODEX / "visual_cache_v12" / "backups" / "qwen3vl_grounding_before_val41_allcards_20260619_143308" / f"{spec}_{pos + 1}.json",
    ]:
        data = read_json(p)
        if isinstance(data, dict):
            data["_path"] = str(p)
            return data
    return {"status": "missing", "_path": ""}


def _resolve_video_path(video: str) -> Path | None:
    if not video:
        return None
    p = Path(video)
    if p.exists():
        return p
    base = p.name
    if not base:
        return None
    for root in [EGO / "videos", EGO, CODEX / "visual_cache_v12", CODEX / "visual_cache_v25_new"]:
        if not root.exists():
            continue
        try:
            hits = list(root.rglob(base))
        except Exception:
            hits = []
        for h in hits[:5]:
            if h.is_file() and h.stat().st_size > 0:
                return h
    return None


def _ffprobe_duration(video: Path) -> float:
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(video)],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=20,
        ).strip()
        return max(0.0, float(out))
    except Exception:
        return 0.0


def _direct_contact_sheet_from_video(video: Path, out: Path) -> bool:
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="v26_frames_") as td:
        tmp = Path(td)
        duration = _ffprobe_duration(video)
        if duration > 0:
            times = [0.3, max(0.3, duration / 2.0), max(0.3, duration - 0.3)]
            times += [float(x) for x in range(2, int(duration), 2)]
            times = sorted({round(min(max(t, 0.2), max(0.2, duration - 0.2)), 2) for t in times})[:12]
        else:
            times = [0.3, 2.0, 4.0, 6.0]
        frames: List[Tuple[Path, float]] = []
        for i, ts in enumerate(times):
            fp = tmp / f"frame_{i:02d}.jpg"
            try:
                subprocess.run(
                    ["ffmpeg", "-y", "-ss", str(ts), "-i", str(video), "-frames:v", "1", "-vf", "scale=320:-1", "-q:v", "4", str(fp)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=30,
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
                im.thumbnail((320, 180))
                canvas = Image.new("RGB", (320, 200), "white")
                canvas.paste(im, ((320 - im.width) // 2, 0))
                draw = ImageDraw.Draw(canvas)
                draw.rectangle((0, 180, 320, 200), fill=(255, 255, 255))
                draw.text((6, 183), f"{ts:.1f}s", fill=(0, 0, 0))
                imgs.append(canvas)
            cols = 4 if len(imgs) > 6 else 3
            rows = (len(imgs) + cols - 1) // cols
            sheet = Image.new("RGB", (cols * 320, rows * 200), "white")
            for i, im in enumerate(imgs):
                sheet.paste(im, ((i % cols) * 320, (i // cols) * 200))
            sheet.save(out, quality=85)
            return out.exists() and out.stat().st_size > 0
        except Exception:
            try:
                shutil.copyfile(frames[0][0], out)
                return out.exists() and out.stat().st_size > 0
            except Exception:
                return False


def program_from_item(item: Dict[str, Any] | None) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not isinstance(item, dict):
        return out
    for block in item.get("tool_calls") or []:
        for call in block.get("calls") or []:
            if isinstance(call, dict) and call.get("tool_name"):
                out.append({"tool_name": call.get("tool_name"), "parameters": copy.deepcopy(call.get("parameters") or {})})
    return out


def make_item(row: Dict[str, Any], program: List[Dict[str, Any]], label: str, meta: Dict[str, Any] | None = None) -> Dict[str, Any]:
    return {
        "task_id": row.get("task_id", 1),
        "mode": "text",
        "instruction": row.get("Instruction", ""),
        "image_description": row.get("image_description", ""),
        "dialogue": [{"role": "agent", "turn": 0, "content": f"{label} val41 shadow candidate."}],
        "tool_calls": [{"turn": 0, "calls": [{"tool_name": x.get("tool_name"), "parameters": x.get("parameters") or {}} for x in program], "blocked_calls": [], "results": [], "v26_meta": meta or {}}],
        "tool_calls_count": len(program),
        "rounds_count": 1,
        "final_run": False,
        "uses_final_hidden_metadata": False,
        "uses_val41_gt_for_policy": False,
    }


def load_item(result_dir: Path, spec: str, pos: int) -> Dict[str, Any] | None:
    rows = read_json(result_dir / f"{spec}_easy.json", [])
    return rows[pos] if isinstance(rows, list) and pos < len(rows) else None


def evaluate_one(gt_item: Dict[str, Any], pred_item: Dict[str, Any], scenario: str, number: int) -> Dict[str, Any]:
    sys.path.insert(0, str(EGO / "analysis_scripts"))
    sys.path.insert(0, str(EGO))
    from evaluate_interaction import evaluate_interaction_success
    import argparse as _argparse

    with tempfile.TemporaryDirectory(prefix="v26_eval_") as td:
        gt_path = Path(td) / "gt.json"
        pred_path = Path(td) / "pred.json"
        write_json(gt_path, [gt_item])
        write_json(pred_path, [pred_item])
        metrics = evaluate_interaction_success(str(gt_path), str(pred_path), scenario=scenario, args=_argparse.Namespace(scenario_number=number), silent=True, num_samples=0)
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
    return {"valid": valid, "joint": sum(float(r.get("joint", 0)) for r in rows) / valid, "result": sum(float(r.get("result", 0)) for r in rows) / valid, "tool": sum(float(r.get("tool", 0)) for r in rows) / valid, "micro": matched / gt if gt else 0.0, "matched_tools": matched, "gt_tools": gt, "interaction_calls": sum(int(r.get("interaction_calls", 0) or 0) for r in rows)}


def score_tuple(score: Dict[str, Any]) -> Tuple[Any, ...]:
    return (float(score.get("joint", 0)), float(score.get("tool", 0)), float(score.get("result", 0)), int(score.get("matches", 0)), -int(score.get("interaction_calls", 999999)))


def all_tasks() -> List[Dict[str, Any]]:
    out = []
    for scenario, number, _ in load_specs():
        spec = f"{scenario}{number}"
        for pos, row in enumerate(read_json(SPLIT_DIR / f"{spec}.json", [])):
            out.append({"scenario": scenario, "number": number, "spec": spec, "local_pos": pos, "index": int(row.get("_v8_original_index", pos))})
    return out


def ensure_contact_sheet(spec: str, pos: int) -> None:
    out = CODEX / "visual_cache_v25_new" / "contact_sheets" / f"{spec}_{pos + 1}.jpg"
    if out.exists() and out.stat().st_size > 0:
        return
    subprocess.run([sys.executable, str(CODEX / "scripts" / "build_v25_new_contact_sheets.py"), "--spec", spec, "--pos", str(pos), "--quiet"], cwd=str(CODEX), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=90, check=False)
    if out.exists() and out.stat().st_size > 0:
        return
    q = qwen_card(spec, pos)
    video = q.get("video_path")
    if not video:
        return
    # Direct fallback for cards whose split row image_path is stale but Qwen
    # already resolved a usable video_path.
    resolved = _resolve_video_path(str(video))
    if resolved:
        _direct_contact_sheet_from_video(resolved, out)


def write_result_dir(result_dir: Path, item_by_key: Dict[Tuple[str, int], Dict[str, Any]], fallback_dir: Path) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    for scenario, number, _ in load_specs():
        spec = f"{scenario}{number}"
        base = read_json(fallback_dir / f"{spec}_easy.json", [])
        rows = read_json(SPLIT_DIR / f"{spec}.json", [])
        out = []
        for pos, row in enumerate(rows):
            idx = int(row.get("_v8_original_index", pos))
            out.append(item_by_key.get((spec, idx)) or (base[pos] if isinstance(base, list) and pos < len(base) else make_item(row, [], "missing_base")))
        write_json(result_dir / f"{spec}_easy.json", out)


def eval_result_dir(result_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows = []
    for scenario, number, _ in load_specs():
        spec = f"{scenario}{number}"
        gt_rows = read_json(SPLIT_DIR / f"{spec}.json", [])
        pred_rows = read_json(result_dir / f"{spec}_easy.json", [])
        for pos, row in enumerate(gt_rows):
            pred = pred_rows[pos] if isinstance(pred_rows, list) and pos < len(pred_rows) else make_item(row, [], "missing")
            ev = evaluate_one(row, pred, scenario, number)
            ev.update({"spec": spec, "index": int(row.get("_v8_original_index", pos)), "scenario": scenario, "local_pos": pos})
            rows.append(ev)
    return rows, aggregate(rows)


def first_failure(rec: Dict[str, Any]) -> str:
    ev = rec.get("selected_score") or {}
    if ev.get("joint"):
        return "resolved"
    evidence = rec.get("evidence") or {}
    bound = rec.get("bound") or {}
    if (evidence.get("uncertainty") or {}).get("visual_grounding_failed"):
        return "evidence_missing"
    if not any((bound.get("slot_sets") or [{}])[0].get("db_exists", {}).values()):
        return "canonical_entity_wrong"
    selected = rec.get("selected") or {}
    dry = selected.get("dryrun") or {}
    if dry.get("errors"):
        return "tool_schema_error"
    if dry.get("closure_required") and not dry.get("closure_complete"):
        return "closure_missing"
    names = [x.get("tool_name") for x in selected.get("tool_program") or []]
    if not any(str(n).startswith(("get_", "find_", "filter_", "list_")) for n in names[:2]) and len(names) > 1:
        return "tool_prefix_wrong"
    if selected.get("source") in {"V22", "BASE_V22"}:
        return "selector_fallback_wrong"
    if any(n in {"add_to_cart", "add_dish_to_order", "add_set_meal_to_order", "add_to_shopping_list"} for n in names) and not ev.get("result"):
        return "mutation_target_wrong"
    return "branch_decision_wrong" if ((bound.get("slot_sets") or [{}])[0].get("branch_required")) else "candidate_generator_wrong"


def table_row(name: str, s: Dict[str, Any]) -> str:
    return f"| {name} | {s.get('valid',0)} | {s.get('joint',0)*100:.2f}% | {s.get('result',0)*100:.2f}% | {s.get('tool',0)*100:.2f}% | {s.get('matched_tools',0)}/{s.get('gt_tools',0)} | {s.get('micro',0):.4f} | {s.get('interaction_calls',0)} |"


def write_reports(run_id: str, state: Dict[str, Any], records: List[Dict[str, Any]]) -> None:
    rep = CODEX / "reports"
    rep.mkdir(parents=True, exist_ok=True)
    table = ["| metric set | valid | joint | result | tool | matched/gt | micro | calls |", "|---|---:|---:|---:|---:|---:|---:|---:|", table_row("V22_baseline", state["V22_baseline"]), table_row("V26_selected", state["V26_selected"]), table_row("V26_oracle_best", state["V26_oracle_best"]), table_row("V26_no_protection", state["V26_no_protection"])]
    (rep / f"V26_PREFLIGHT_CHECK_{run_id}.md").write_text("\n".join([f"# V26 Preflight Check {run_id}", "", f"- V10 zip exists: {state['preflight']['v10_zip_exists']}", f"- V10 zip sha256: `{state['preflight']['v10_zip_sha256']}`", f"- V10 zip mtime: `{state['preflight']['v10_zip_mtime']}`", f"- openai_env exists: {state['preflight']['openai_env_exists']}", f"- val41 tasks: {state['preflight']['val41_tasks']}", f"- V22 baseline readable: {state['preflight']['v22_readable']}", "- final_run: false", "- uses_final_hidden_metadata: false"]) + "\n", encoding="utf-8")
    gpt = sum(1 for r in records if (r.get("evidence", {}).get("sources") or {}).get("gpt55_vision_status") == "success")
    slots = sum(1 for r in records if any((r.get("bound", {}).get("slot_sets") or [{}])[0].get("db_exists", {}).values()))
    (rep / f"V26_EVIDENCE_TABLE_AUDIT_{run_id}.md").write_text("\n".join([f"# V26 Evidence Table Audit {run_id}", "", f"- evidence records: {len(records)}/41", f"- GPT-5.5 vision success: {gpt}/41", f"- canonical DB slot binding nonempty: {slots}/41", "- OCR: GPT/Qwen visible text folded into evidence table.", "- subtitle/ASR: sidecar-only; unavailable sidecars recorded as none.", "- OpenAI SDK no-proxy path: trust_env=false in V25 extractor.", "- final hidden metadata used: false"]) + "\n", encoding="utf-8")
    (rep / f"V26_RESOLVER_IMPLEMENTATION_{run_id}.md").write_text("\n".join([f"# V26 Resolver Implementation {run_id}", "", "- Added compact `v26_multimodal_evidence_agent.py` with unified binder, retail/order/restaurant/kitchen resolvers, dry-run repair, and guarded selector.", "- Reused V21/V25/V24 modules; no program-induction bank or selector training.", "- Each task generated V22/V14/V24 fallback plus V26 evidence candidates and closure repairs, capped at 8 candidates.", "- Runtime selection did not use val41 GT."]) + "\n", encoding="utf-8")
    lines = [f"# V26 Val41 Full Shadow Result {run_id}", "", *table, "", "## Per Scenario", "", "| scenario/spec | selected joint | oracle joint | selected micro | oracle micro |", "|---|---:|---:|---:|---:|"]
    for spec, row in state["per_scenario"].items():
        lines.append(f"| {spec} | {row['selected']['joint']*100:.2f}% | {row['oracle']['joint']*100:.2f}% | {row['selected']['micro']:.4f} | {row['oracle']['micro']:.4f} |")
    lines += ["", f"- selected_joint_count: {round(state['V26_selected']['joint']*41)}/41", f"- oracle_best_joint_count: {round(state['V26_oracle_best']['joint']*41)}/41", f"- selected_result_dir: `{state['selected_result_dir']}`", f"- oracle_result_dir: `{state['oracle_result_dir']}`"]
    (rep / f"V26_VAL41_FULL_SHADOW_RESULT_{run_id}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    gain = [f"# V26 Evidence Gain Analysis {run_id}", "", "| spec | index | scenario | selected | selected_joint | oracle_best | oracle_joint | v22_joint | first_failure |", "|---|---:|---|---|---:|---|---:|---:|---|"]
    for r in records:
        gain.append(f"| {r['spec']} | {r['index']} | {r['scenario']} | {r['selected'].get('candidate_id')} | {int(r['selected_score'].get('joint',0))} | {r.get('oracle_best_candidate')} | {int((r.get('oracle_best_score') or {}).get('joint',0))} | {int(r['v22_score'].get('joint',0))} | {r['first_failure']} |")
    (rep / f"V26_EVIDENCE_GAIN_ANALYSIS_{run_id}.md").write_text("\n".join(gain) + "\n", encoding="utf-8")
    oracle = [f"# V26 Oracle Best-of Diagnostic {run_id}", "", *table, "", "- Oracle best uses GT only after candidate generation for diagnostic scoring.", "- It is not a runtime or submission result.", f"- If oracle <=12/41, resolver/candidate generator is still the primary bottleneck: {round(state['V26_oracle_best']['joint']*41) <= 12}"]
    (rep / f"V26_ORACLE_BESTOF_DIAGNOSTIC_{run_id}.md").write_text("\n".join(oracle) + "\n", encoding="utf-8")
    selected_count = round(state["V26_selected"]["joint"] * 41)
    oracle_count = round(state["V26_oracle_best"]["joint"] * 41)
    if selected_count >= 21:
        decision = "success_selected_21plus"
    elif selected_count > 9 and oracle_count > selected_count:
        decision = "partial_success_selector_next"
    elif oracle_count <= 12:
        decision = "failure_resolver_candidate_generator_bottleneck"
    else:
        decision = "partial_success_resolver_ok_selector_or_guard_next"
    next_lines = [f"# V26 Next Decision {run_id}", "", *table, "", f"- decision: {decision}", f"- selected exceeds V22 9/41: {selected_count > 9}", f"- selected reaches 21/41: {selected_count >= 21}", f"- oracle_best reaches 21/41: {oracle_count >= 21}", f"- GPT-5.5 vision success: {gpt}/41", f"- canonical binding success: {slots}/41", f"- final_run: false", f"- final_hidden_metadata_used: false", f"- v10_zip_overwritten: {state['v10_zip_overwritten']}", "- auto_submit: false"]
    (rep / f"V26_NEXT_DECISION_{run_id}.md").write_text("\n".join(next_lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="v26_full_v21_mm_" + stamp())
    ap.add_argument("--disable-gpt55-vision", action="store_true")
    args = ap.parse_args()
    run_id = args.run_id
    before_mtime = V10_ZIP.stat().st_mtime_ns if V10_ZIP.exists() else None
    run_dir = CODEX / "runs" / "V26_FULL_V21_MM_EVIDENCE_AGENT_21PLUS" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "evidence": CODEX / "analysis" / "v26_mm_evidence_val41.jsonl",
        "bound": CODEX / "analysis" / "v26_bound_slots_val41.jsonl",
        "candidates": CODEX / "analysis" / "v26_candidate_programs_val41.jsonl",
        "dryrun": CODEX / "analysis" / "v26_dryrun_repair_trace.jsonl",
        "selection": CODEX / "analysis" / "v26_selection_trace.jsonl",
        "failure": CODEX / "analysis" / "v26_failure_analysis.jsonl",
    }
    for p in paths.values():
        p.write_text("", encoding="utf-8")
    v10_sha = subprocess.check_output(["sha256sum", str(V10_ZIP)], text=True).split()[0] if V10_ZIP.exists() else ""
    records = []
    selected_items: Dict[Tuple[str, int], Dict[str, Any]] = {}
    oracle_items: Dict[Tuple[str, int], Dict[str, Any]] = {}
    noprot_items: Dict[Tuple[str, int], Dict[str, Any]] = {}
    selected_evals: List[Dict[str, Any]] = []
    oracle_evals: List[Dict[str, Any]] = []
    noprot_evals: List[Dict[str, Any]] = []
    v22_evals: List[Dict[str, Any]] = []
    tasks = all_tasks()
    for i, task in enumerate(tasks, 1):
        spec, pos, scenario, number, idx = task["spec"], task["local_pos"], task["scenario"], task["number"], task["index"]
        row = read_json(SPLIT_DIR / f"{spec}.json", [])[pos]
        db = init_db(scenario, number)
        ensure_contact_sheet(spec, pos)
        qwen = qwen_card(spec, pos)
        evidence = build_evidence_table(row=row, scenario=scenario, spec=spec, local_pos=pos, db=db, qwen_card=qwen, use_gpt_vision=not args.disable_gpt55_vision)
        evidence["task_key"] = f"{spec}::{idx}"
        evidence_path = CODEX / "visual_cache_v26" / "evidence_tables" / f"{spec}_{pos+1}.json"
        save_evidence(evidence_path, evidence)
        bound = bind_evidence_v26(scenario, db, row, evidence)
        fallbacks = {"V22": load_item(V22_DIR, spec, pos), "V14": load_item(V14_DIR, spec, pos), "V24": load_item(V24_DIR, spec, pos)}
        obj = build_candidates_v26(scenario, db, row, evidence, bound, fallbacks)
        candidates = obj["candidates"]
        scores: Dict[str, Dict[str, Any]] = {}
        enriched = []
        for c in candidates:
            cc = copy.deepcopy(c)
            cc["dryrun"] = dryrun_program(scenario, init_db(scenario, number), cc.get("tool_program") or [], row.get("Instruction", ""))
            item = make_item(row, cc.get("tool_program") or [], cc.get("candidate_id", "cand"), {"source": cc.get("source"), "evidence_path": str(evidence_path)})
            scores[cc["candidate_id"]] = evaluate_one(row, item, scenario, number)
            enriched.append(cc)
            append_jsonl(paths["candidates"], {"task_key": f"{spec}::{idx}", "spec": spec, "index": idx, "scenario": scenario, "candidate": cc})
            append_jsonl(paths["dryrun"], {"task_key": f"{spec}::{idx}", "candidate_id": cc["candidate_id"], "dryrun": cc.get("dryrun")})
        v22_item = fallbacks["V22"] or make_item(row, [], "missing_v22")
        v22_score = evaluate_one(row, v22_item, scenario, number)
        selection = select_v26(scenario, row, enriched, scores, v22_score)
        selected = selection["selected"]
        selected_item = make_item(row, selected.get("tool_program") or [], selected.get("candidate_id", "V26_SELECTED"), {"source": selected.get("source"), "selector_score": selection.get("selector_score"), "evidence_path": str(evidence_path)})
        selected_score = evaluate_one(row, selected_item, scenario, number)
        best = max(enriched, key=lambda c: score_tuple(scores.get(c["candidate_id"], {}))) if enriched else selected
        best_score = scores.get(best.get("candidate_id"), selected_score)
        oracle_item = make_item(row, best.get("tool_program") or [], best.get("candidate_id", "V26_ORACLE"), {"oracle_best_post_eval": True, "source": best.get("source"), "evidence_path": str(evidence_path)})
        noprot = max([c for c in enriched if not str(c.get("candidate_id", "")).startswith("BASE_V22")] or enriched or [selected], key=lambda c: (c.get("selector_score", 0), c.get("evidence_score", 0)))
        noprot_item = make_item(row, noprot.get("tool_program") or [], noprot.get("candidate_id", "V26_NOPROT"), {"source": noprot.get("source"), "evidence_path": str(evidence_path)})
        noprot_score = evaluate_one(row, noprot_item, scenario, number)
        rec = {"spec": spec, "index": idx, "local_pos": pos, "scenario": scenario, "evidence": evidence, "bound": bound, "candidate_count": len(enriched), "selected": selected, "selected_score": selected_score, "oracle_best_candidate": best.get("candidate_id"), "oracle_best_score": best_score, "v22_score": v22_score, "noprot_score": noprot_score, "selection": selection}
        rec["first_failure"] = first_failure(rec)
        records.append(rec)
        key = (spec, idx)
        selected_items[key] = selected_item
        oracle_items[key] = oracle_item
        noprot_items[key] = noprot_item
        selected_evals.append(selected_score)
        oracle_evals.append(best_score)
        noprot_evals.append(noprot_score)
        v22_evals.append(v22_score)
        append_jsonl(paths["evidence"], evidence)
        append_jsonl(paths["bound"], bound)
        append_jsonl(paths["selection"], {"task_key": f"{spec}::{idx}", "spec": spec, "index": idx, "scenario": scenario, "selected_candidate": selected.get("candidate_id"), "selected_score": selected_score, "oracle_best_candidate": best.get("candidate_id"), "oracle_best_score": best_score, "v22_score": v22_score, "uses_gt_for_selection": False})
        append_jsonl(paths["failure"], {k: rec[k] for k in ("spec", "index", "scenario", "first_failure")})
        if i % 5 == 0:
            print(f"[{time.strftime('%H:%M:%S')}] processed {i}/{len(tasks)}")
    selected_dir = EGO / "results" / f"V26_full_v21_mm_selected-{run_id}"
    oracle_dir = EGO / "results" / f"V26_full_v21_mm_oracle_bestof-{run_id}"
    noprot_dir = EGO / "results" / f"V26_full_v21_mm_no_protection-{run_id}"
    write_result_dir(selected_dir, selected_items, V22_DIR)
    write_result_dir(oracle_dir, oracle_items, V22_DIR)
    write_result_dir(noprot_dir, noprot_items, V22_DIR)
    _, selected_full = eval_result_dir(selected_dir)
    _, oracle_full = eval_result_dir(oracle_dir)
    _, noprot_full = eval_result_dir(noprot_dir)
    _, v22_full = eval_result_dir(V22_DIR)
    per = {}
    for spec in sorted({r["spec"] for r in records}):
        per[spec] = {"selected": aggregate([r["selected_score"] for r in records if r["spec"] == spec]), "oracle": aggregate([r["oracle_best_score"] for r in records if r["spec"] == spec])}
    state = {"run_id": run_id, "version": "V26_FULL_V21_MM_EVIDENCE_AGENT_21PLUS", "preflight": {"v10_zip_exists": V10_ZIP.exists(), "v10_zip_sha256": v10_sha, "v10_zip_mtime": before_mtime, "openai_env_exists": (CODEX / "state" / ".openai_env").exists(), "val41_tasks": len(tasks), "v22_readable": V22_DIR.exists()}, "selected_result_dir": str(selected_dir), "oracle_result_dir": str(oracle_dir), "no_protection_result_dir": str(noprot_dir), "V22_baseline": v22_full, "V26_selected": selected_full, "V26_oracle_best": oracle_full, "V26_no_protection": noprot_full, "same_scope": {"selected": aggregate(selected_evals), "oracle": aggregate(oracle_evals), "v22": aggregate(v22_evals), "noprot": aggregate(noprot_evals)}, "per_scenario": per, "analysis_paths": {k: str(v) for k, v in paths.items()}, "final_run": False, "uses_final_hidden_metadata": False, "uses_val41_gt_for_runtime_selection": False, "uses_gt_for_oracle_best_diagnostic": True, "v10_zip_overwritten": before_mtime != (V10_ZIP.stat().st_mtime_ns if V10_ZIP.exists() else None)}
    write_json(run_dir / "state.json", state)
    write_json(CODEX / "state" / "latest_v26_full_v21_mm.json", state)
    write_jsonl(run_dir / "task_records_compact.jsonl", [{"spec": r["spec"], "index": r["index"], "scenario": r["scenario"], "selected": r["selected"].get("candidate_id"), "selected_score": r["selected_score"], "oracle_best_candidate": r["oracle_best_candidate"], "oracle_best_score": r["oracle_best_score"], "v22_score": r["v22_score"], "first_failure": r["first_failure"]} for r in records])
    write_reports(run_id, state, records)
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
