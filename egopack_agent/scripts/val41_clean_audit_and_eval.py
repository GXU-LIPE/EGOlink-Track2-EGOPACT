#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit frozen val41 task hygiene and evaluate versions on clean subset only.

This script does not train, distill, or read final309 hidden metadata. It uses
val41 labels only for auditing/evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


CODEX = Path(os.environ.get("CODEX_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/codex"))
EGO = Path(os.environ.get("EGO_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/EgoBench"))
SPLIT = CODEX / "state" / "materialized_splits" / "validation_A_limit30"
VIDEOS = EGO / "videos"


USER_ID_RE = re.compile(
    r"(?:user\s*id|user_id|customer\s*id|customer_id)\s*(?:is|=|:)?\s*['\"]?([A-Za-z][A-Za-z0-9_]*\d[A-Za-z0-9_]*)",
    re.I,
)
GENERIC_ID_RE = re.compile(r"\b(?:user|customer|cook|grace|manager|nutritionist|critic|enthusiast)_[A-Za-z0-9_]*\d[A-Za-z0-9_]*\b", re.I)

CONDITION_KEYWORDS = {
    "price": ["price", "cheapest", "expensive", "cost", "payable", "payment", "amount"],
    "tax": ["tax"],
    "discount": ["discount"],
    "nutrition": ["nutrition", "calorie", "protein", "fat", "carb", "sugar", "sodium", "fiber"],
    "taste": ["taste", "sour", "bitter", "sweet", "buttery", "aroma"],
    "origin": ["country", "origin", "france", "italy", "imported"],
    "category": ["category", "section", "shelf"],
    "allergen": ["milk", "allergen", "contains"],
    "quantity": ["quantity", "stock", "amount"],
}

GT_TOOL_HINTS = {
    "price": ["price", "payment"],
    "tax": ["tax"],
    "discount": ["discount"],
    "nutrition": ["nutrition"],
    "taste": ["taste", "profile"],
    "origin": ["country", "origin"],
    "category": ["category"],
    "allergen": ["allergen", "ingredient", "nutrition"],
}

ENTITY_KEYS = (
    "product_name",
    "dish_name",
    "set_meal_name",
    "ingredient_name",
    "recipe_name",
    "category",
    "restaurant_name",
)


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return default


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def canonical(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s&'éèàùçöäü-]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_instruction_user_ids(text: str) -> List[str]:
    out: List[str] = []
    for m in USER_ID_RE.findall(text or ""):
        if m not in out:
            out.append(m)
    for m in GENERIC_ID_RE.findall(text or ""):
        if m not in out:
            out.append(m)
    return out


def walk(obj: Any) -> Iterable[Any]:
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from walk(v)
    elif isinstance(obj, list):
        for item in obj:
            yield from walk(item)


def gt_user_ids(gt: List[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    for d in walk(gt):
        if not isinstance(d, dict):
            continue
        for key in ("user_id", "customer_id"):
            val = d.get(key)
            if val and str(val) not in out:
                out.append(str(val))
    return out


def gt_entities(gt: List[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    for d in walk(gt):
        if not isinstance(d, dict):
            continue
        for key in ENTITY_KEYS:
            val = d.get(key)
            if isinstance(val, str) and val and canonical(val) not in {canonical(x) for x in out}:
                out.append(val)
    return out


def condition_flags(text: str) -> Dict[str, bool]:
    t = (text or "").lower()
    return {name: any(k in t for k in kws) for name, kws in CONDITION_KEYWORDS.items()}


def tool_names(gt: List[Dict[str, Any]]) -> List[str]:
    return [str(c.get("tool_name", "")) for c in gt if isinstance(c, dict)]


def gt_supports_condition(cond: str, tools: List[str], gt_text: str) -> bool:
    joined = " ".join(tools).lower() + " " + gt_text.lower()
    return any(h in joined for h in GT_TOOL_HINTS.get(cond, [cond]))


def value_entities(row: Dict[str, Any]) -> List[str]:
    vals = row.get("value")
    if isinstance(vals, list):
        return [str(v) for v in vals if isinstance(v, (str, int, float))]
    if isinstance(vals, (str, int, float)):
        return [str(vals)]
    return []


def video_ok(spec: str, row: Dict[str, Any]) -> Tuple[bool, str]:
    path = row.get("video_path") or row.get("image_path") or row.get("image_name") or ""
    if not path:
        return False, "missing_video_path"
    base = Path(str(path)).name
    stem = Path(base).stem
    if stem != spec:
        return False, f"basename_{stem}_not_{spec}"
    if not (VIDEOS / base).exists():
        return False, f"video_missing:{VIDEOS / base}"
    return True, "ok"


def materialized_index_ok(spec: str, row: Dict[str, Any], local_pos: int, manifest_indices: List[int]) -> Tuple[bool, str]:
    """Verify the materialized row still points to the manifest source index.

    Do not open EgoBench scenarios/final here. The clean audit is allowed to
    use frozen val41 labels for filtering/evaluation, but not final hidden
    metadata. The materialized split manifest is the safe source of row identity.
    """
    orig = row.get("_v8_original_index")
    if orig is None:
        return False, "missing_original_index"
    try:
        oi = int(orig)
    except Exception:
        return False, f"bad_original_index:{orig}"
    if local_pos >= len(manifest_indices):
        return False, f"local_pos_{local_pos}_not_in_manifest"
    expected = int(manifest_indices[local_pos])
    if oi != expected:
        return False, f"manifest_index_mismatch:row={oi}:manifest={expected}"
    if not row.get("Instruction"):
        return False, "missing_instruction"
    if not row.get("ground_truth"):
        return False, "missing_ground_truth"
    return True, f"ok_manifest_index:{expected}"


def build_oracle_result(row: Dict[str, Any], task_id: int) -> Dict[str, Any]:
    calls = row.get("ground_truth") or []
    return {
        "task_id": task_id,
        "mode": "text",
        "instruction": row.get("Instruction", ""),
        "image_description": row.get("image_description", ""),
        "dialogue": [{"role": "agent", "turn": 0, "content": "val41 clean audit oracle replay"}],
        "tool_calls": [{"turn": 0, "calls": calls, "blocked_calls": [], "results": []}],
        "tool_calls_count": len(calls),
        "rounds_count": 1,
        "input_tokens": 0,
        "output_tokens": 0,
        "tokens_consumed": 0,
        "final_run": False,
    }


def evaluate_spec(gt_path: Path, pred_path: Path, scenario: str, number: int) -> Dict[str, Any]:
    sys.path.insert(0, str(EGO / "analysis_scripts"))
    sys.path.insert(0, str(EGO))
    from evaluate_interaction import evaluate_interaction_success
    import argparse as _argparse

    metrics = evaluate_interaction_success(
        str(gt_path),
        str(pred_path),
        scenario=scenario,
        args=_argparse.Namespace(scenario_number=number),
        silent=True,
        num_samples=0,
    )
    micro = metrics.get("micro_tool_stats", {}) or {}
    return {
        "valid": metrics.get("valid_scenarios", 0),
        "joint": metrics.get("joint_success", {}).get("success_rate", 0),
        "result": metrics.get("result_based", {}).get("success_rate", 0),
        "tool": metrics.get("tool_based", {}).get("success_rate", 0),
        "micro": micro.get("micro_accuracy", 0),
        "correct_calls": micro.get("total_correct_calls", 0),
        "gt_calls": micro.get("total_ground_truth_calls", 0),
        "interaction_calls": micro.get("total_interaction_calls", 0),
    }


def replay_gt(spec: str, scenario: str, number: int, row: Dict[str, Any], run_id: str) -> Tuple[bool, Dict[str, Any]]:
    replay_dir = CODEX / "runs" / "VAL41_CLEAN_AUDIT" / run_id / "gt_replay"
    gt_path = replay_dir / "gt" / f"{spec}_{row.get('_v8_original_index')}.json"
    pred_path = replay_dir / "pred" / f"{spec}_{row.get('_v8_original_index')}_easy.json"
    write_json(gt_path, [row])
    write_json(pred_path, [build_oracle_result(row, 1)])
    try:
        metrics = evaluate_spec(gt_path, pred_path, scenario, number)
        ok = metrics.get("valid") == 1 and metrics.get("joint", 0) >= 0.999 and metrics.get("tool", 0) >= 0.999
        return ok, metrics
    except Exception as exc:
        return False, {"error": f"{type(exc).__name__}: {exc}"}


def audit_task(spec: str, scenario: str, number: int, local_pos: int, row: Dict[str, Any], run_id: str, manifest_indices: List[int]) -> Dict[str, Any]:
    instruction = row.get("Instruction") or row.get("instruction") or ""
    analysis = row.get("analysis") or ""
    gt = row.get("ground_truth") or []
    gt_text = json.dumps(gt, ensure_ascii=False)
    labels: List[str] = []
    reasons: List[str] = []

    inst_ids = extract_instruction_user_ids(instruction)
    gt_ids = gt_user_ids(gt)
    if inst_ids or gt_ids:
        if sorted(inst_ids) != sorted(gt_ids):
            labels.append("suspicious_user_mismatch")
            reasons.append(f"instruction_user_ids={inst_ids} gt_tool_user_ids={gt_ids}")

    inst_flags = condition_flags(instruction)
    ana_flags = condition_flags(analysis)
    tools = tool_names(gt)
    for cond, present in inst_flags.items():
        if not present:
            continue
        if cond in {"category", "quantity"}:
            continue
        if not ana_flags.get(cond) and not gt_supports_condition(cond, tools, gt_text):
            labels.append("suspicious_condition_mismatch")
            reasons.append(f"instruction_condition_{cond}_not_supported_by_analysis_or_gt")
    for cond, present in ana_flags.items():
        if present and cond not in {"category", "quantity"} and not inst_flags.get(cond) and gt_supports_condition(cond, tools, gt_text):
            labels.append("suspicious_condition_mismatch")
            reasons.append(f"analysis_gt_condition_{cond}_not_in_instruction")
    if ("add" in instruction.lower() or "remove" in instruction.lower()) and not any(n.startswith(("add", "remove")) for n in tools):
        labels.append("suspicious_condition_mismatch")
        reasons.append("instruction_requests_mutation_but_gt_has_no_mutation")

    vals = value_entities(row)
    gt_ents = gt_entities(gt)
    if vals and gt_ents:
        vals_c = {canonical(v) for v in vals}
        gt_c = {canonical(v) for v in gt_ents}
        # Only require overlap for same entity-bearing row; avoid false positive on
        # restaurant/category helper fields by checking product/dish-like keys.
        if str(row.get("key", "")).lower() in ENTITY_KEYS and not (vals_c & gt_c):
            labels.append("suspicious_entity_mismatch")
            reasons.append(f"value_entities={vals} gt_entities={gt_ents}")

    vok, vreason = video_ok(spec, row)
    if not vok:
        labels.append("suspicious_video_mismatch")
        reasons.append(vreason)

    mok, mreason = materialized_index_ok(spec, row, local_pos, manifest_indices)
    if not mok:
        labels.append("suspicious_entity_mismatch")
        reasons.append(f"materialized_index_failed:{mreason}")

    replay_ok, replay_metrics = replay_gt(spec, scenario, number, row, run_id)
    if not replay_ok:
        labels.append("gt_replay_failed")
        reasons.append(f"gt_replay_failed:{replay_metrics}")

    labels = sorted(set(labels))
    if not labels:
        labels = ["clean"]
    return {
        "uid": f"{spec}::{row.get('_v8_original_index', local_pos + 1)}",
        "spec": spec,
        "scenario": scenario,
        "number": number,
        "local_pos": local_pos,
        "task_id": row.get("task_id"),
        "_v8_original_index": row.get("_v8_original_index"),
        "labels": labels,
        "reasons": reasons,
        "instruction_user_ids": inst_ids,
        "gt_tool_user_ids": gt_ids,
        "instruction_conditions": {k: v for k, v in inst_flags.items() if v},
        "analysis_conditions": {k: v for k, v in ana_flags.items() if v},
        "gt_tool_names": tools,
        "value_entities": vals,
        "gt_entities": gt_ents,
        "video_check": {"ok": vok, "reason": vreason},
        "materialized_index_check": {"ok": mok, "reason": mreason},
        "gt_replay": {"ok": replay_ok, "metrics": replay_metrics},
    }


def load_manifest_specs() -> List[Tuple[str, int, List[int]]]:
    manifest = read_json(SPLIT / "manifest.json", {})
    return [(str(s), int(n), [int(x) for x in idxs]) for s, n, idxs in manifest.get("specs", [])]


def build_clean_split(run_id: str, audit_rows: List[Dict[str, Any]]) -> Tuple[Path, Dict[str, Any]]:
    clean_dir = CODEX / "state" / "materialized_splits" / f"validation_A_clean_{run_id}"
    clean_dir.mkdir(parents=True, exist_ok=True)
    clean_by_spec: Dict[str, List[int]] = {}
    for row in audit_rows:
        if row["labels"] == ["clean"]:
            clean_by_spec.setdefault(row["spec"], []).append(int(row["local_pos"]))
    files = []
    for spec, positions in sorted(clean_by_spec.items()):
        src = read_json(SPLIT / f"{spec}.json", [])
        subset = [src[pos] for pos in positions]
        write_json(clean_dir / f"{spec}.json", subset)
        scenario = "".join(ch for ch in spec if not ch.isdigit())
        number = int(spec[len(scenario):])
        files.append({
            "file": f"{spec}.json",
            "scenario": scenario,
            "number": number,
            "source_local_positions": positions,
            "source_original_indices": [subset[i].get("_v8_original_index") for i in range(len(subset))],
            "task_count": len(subset),
        })
    manifest = {
        "split_name": "validation_A_clean",
        "run_id": run_id,
        "source_split": str(SPLIT),
        "files": files,
        "specs": [(f["scenario"], f["number"], f["source_original_indices"]) for f in files],
        "planned_task_count": sum(f["task_count"] for f in files),
        "uses_val41_gt_for_filtering": True,
        "uses_final_hidden_metadata": False,
    }
    write_json(clean_dir / "manifest.json", manifest)
    return clean_dir, manifest


def find_result_file(version: str, spec: str) -> Optional[Path]:
    candidates = {
        "V10": [
            EGO / "results" / "gpt-5.5-V10_full_memory_final_candidate_draft_A_medium_sanity_20260618_1716" / f"{spec}_easy.json",
        ],
        "V14": [
            EGO / "results" / "gpt-5.5-V14_val41_distilled_no_task_oracle-v14_distilled_val41_20260619_211502" / f"{spec}_easy.json",
        ],
        "V14_candidate": [
            EGO / "results" / "V14_candidate_selection_val41-v14_candidate_selection_20260619_2134" / f"{spec}_easy.json",
            EGO / "results" / "gpt-5.5-V14_candidate_selection_val41-v14_candidate_selection_20260619_2134" / f"{spec}_easy.json",
        ],
        "V17": [
            EGO / "results" / f"gpt-5.5-V17_compiler_repaired_smoke5-v17_smoke5_20260620_1140" / f"{spec}_easy.json",
        ],
    }
    for p in candidates.get(version, []):
        if p.exists():
            return p
    # Fallback search scoped to results only.
    matches = sorted((EGO / "results").glob(f"*{version}*/{spec}_easy.json"))
    return matches[0] if matches else None


def filter_result(src_result: Path, src_spec_file: Path, clean_file: Path, out_path: Path) -> Tuple[bool, str]:
    src_rows = read_json(src_spec_file, [])
    clean_rows = read_json(clean_file, [])
    pred_rows = read_json(src_result, [])
    if len(pred_rows) != len(src_rows):
        return False, f"result_len_{len(pred_rows)}_source_len_{len(src_rows)}"
    selected = []
    for clean_row in clean_rows:
        orig = clean_row.get("_v8_original_index")
        matches = [i for i, src_row in enumerate(src_rows) if src_row.get("_v8_original_index") == orig and src_row.get("Instruction") == clean_row.get("Instruction")]
        if not matches:
            return False, f"clean_row_orig_{orig}_not_found_in_source"
        selected.append(pred_rows[matches[0]])
    write_json(out_path, selected)
    return True, "ok"


def evaluate_clean_existing(run_id: str, clean_dir: Path, manifest: Dict[str, Any], versions: List[str]) -> Dict[str, Any]:
    rows = []
    pred_root = CODEX / "runs" / "VAL41_CLEAN_AUDIT" / run_id / "filtered_predictions"
    for version in versions:
        for file_info in manifest.get("files", []):
            spec = Path(file_info["file"]).stem
            scenario = file_info["scenario"]
            number = int(file_info["number"])
            clean_file = clean_dir / file_info["file"]
            src_spec_file = SPLIT / file_info["file"]
            src_result = find_result_file(version, spec)
            if not src_result:
                rows.append({"version": version, "spec": spec, "valid": 0, "error": "missing_result_file"})
                continue
            out_pred = pred_root / version / f"{spec}_easy.json"
            ok, reason = filter_result(src_result, src_spec_file, clean_file, out_pred)
            if not ok:
                rows.append({"version": version, "spec": spec, "valid": 0, "error": reason, "result_file": str(src_result)})
                continue
            try:
                metrics = evaluate_spec(clean_file, out_pred, scenario, number)
                rows.append({"version": version, "spec": spec, **metrics, "result_file": str(src_result), "error": ""})
            except Exception as exc:
                rows.append({"version": version, "spec": spec, "valid": 0, "error": f"{type(exc).__name__}: {exc}", "result_file": str(src_result)})
    summaries = {}
    for version in versions:
        vr = [r for r in rows if r["version"] == version]
        valid = sum(r.get("valid", 0) for r in vr)
        correct = sum(r.get("correct_calls", 0) for r in vr)
        gt_calls = sum(r.get("gt_calls", 0) for r in vr)
        def wavg(k: str) -> float:
            return sum(r.get(k, 0) * r.get("valid", 0) for r in vr) / valid if valid else 0.0
        summaries[version] = {
            "valid": valid,
            "joint": wavg("joint"),
            "result": wavg("result"),
            "tool": wavg("tool"),
            "micro": correct / gt_calls if gt_calls else wavg("micro"),
            "correct_calls": correct,
            "gt_calls": gt_calls,
            "interaction_calls": sum(r.get("interaction_calls", 0) for r in vr),
            "missing_or_error_specs": [r for r in vr if r.get("error")],
        }
    return {"rows": rows, "summary": summaries}


def write_reports(run_id: str, audit_rows: List[Dict[str, Any]], clean_manifest: Dict[str, Any], eval_result: Dict[str, Any]) -> Tuple[Path, Path]:
    audit_report = CODEX / "reports" / f"VAL41_CLEAN_AUDIT_{run_id}.md"
    counts: Dict[str, int] = {}
    for row in audit_rows:
        for label in row["labels"]:
            counts[label] = counts.get(label, 0) + 1
    by_spec: Dict[str, Dict[str, int]] = {}
    for row in audit_rows:
        d = by_spec.setdefault(row["spec"], {"total": 0, "clean": 0, "suspicious": 0})
        d["total"] += 1
        if row["labels"] == ["clean"]:
            d["clean"] += 1
        else:
            d["suspicious"] += 1
    lines = [
        f"# Val41 Clean Audit {run_id}",
        "",
        "- final_run: false",
        "- uses_val41_gt_for_filtering_and_eval: true",
        "- uses_final_hidden_metadata: false",
        f"- total_tasks: {len(audit_rows)}",
        f"- clean_tasks: {counts.get('clean', 0)}",
        "",
        "## Label Counts",
        "",
    ]
    for k, v in sorted(counts.items()):
        lines.append(f"- {k}: {v}")
    lines += ["", "## Per Spec", "", "| spec | total | clean | suspicious |", "|---|---:|---:|---:|"]
    for spec, d in sorted(by_spec.items()):
        lines.append(f"| {spec} | {d['total']} | {d['clean']} | {d['suspicious']} |")
    lines += ["", "## Suspicious Tasks", "", "| uid | labels | reasons |", "|---|---|---|"]
    for row in audit_rows:
        if row["labels"] != ["clean"]:
            lines.append(f"| {row['uid']} | {', '.join(row['labels'])} | {'; '.join(row['reasons'])[:500]} |")
    audit_report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    eval_report = CODEX / "reports" / f"VAL41_CLEAN_EVAL_{run_id}.md"
    lines = [
        f"# Val41 Clean-Only Evaluation {run_id}",
        "",
        "- final_run: false",
        "- evaluated_subset: clean only",
        f"- clean_tasks: {clean_manifest.get('planned_task_count')}",
        f"- clean_split: `{CODEX / 'state' / 'materialized_splits' / ('validation_A_clean_' + run_id)}`",
        "",
        "## Summary",
        "",
        "| version | valid | joint | result | tool | micro | calls | interaction_calls | errors |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for version, s in sorted(eval_result["summary"].items()):
        lines.append(f"| {version} | {s.get('valid', 0)} | {s.get('joint', 0):.4f} | {s.get('result', 0):.4f} | {s.get('tool', 0):.4f} | {s.get('micro', 0):.4f} | {s.get('correct_calls', 0)}/{s.get('gt_calls', 0)} | {s.get('interaction_calls', 0)} | {len(s.get('missing_or_error_specs', []))} |")
    lines += ["", "## Per Spec", "", "| version | spec | valid | joint | result | tool | micro | calls | error |", "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for row in sorted(eval_result["rows"], key=lambda r: (r["version"], r["spec"])):
        lines.append(f"| {row['version']} | {row['spec']} | {row.get('valid', 0)} | {row.get('joint', 0):.3f} | {row.get('result', 0):.3f} | {row.get('tool', 0):.3f} | {row.get('micro', 0):.3f} | {row.get('correct_calls', 0)}/{row.get('gt_calls', 0)} | {row.get('error', '')} |")
    eval_report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return audit_report, eval_report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default=f"val41_clean_{time.strftime('%Y%m%d_%H%M%S')}")
    ap.add_argument("--versions", default="V10,V14,V14_candidate,V17")
    args = ap.parse_args()
    run_id = args.run_id
    audit_rows: List[Dict[str, Any]] = []
    for scenario, number, idxs in load_manifest_specs():
        spec = f"{scenario}{number}"
        rows = read_json(SPLIT / f"{spec}.json", [])
        for pos, row in enumerate(rows):
            audit_rows.append(audit_task(spec, scenario, number, pos, row, run_id, idxs))
    out_dir = CODEX / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_jsonl = out_dir / f"VAL41_CLEAN_AUDIT_{run_id}.jsonl"
    with audit_jsonl.open("w", encoding="utf-8") as f:
        for row in audit_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    clean_dir, clean_manifest = build_clean_split(run_id, audit_rows)
    eval_result = evaluate_clean_existing(run_id, clean_dir, clean_manifest, [v.strip() for v in args.versions.split(",") if v.strip()])
    eval_json = out_dir / f"VAL41_CLEAN_EVAL_{run_id}.json"
    write_json(eval_json, eval_result)
    audit_report, eval_report = write_reports(run_id, audit_rows, clean_manifest, eval_result)
    state = {
        "run_id": run_id,
        "audit_jsonl": str(audit_jsonl),
        "eval_json": str(eval_json),
        "clean_split": str(clean_dir),
        "audit_report": str(audit_report),
        "eval_report": str(eval_report),
        "clean_tasks": clean_manifest.get("planned_task_count"),
        "summary": eval_result.get("summary"),
        "final_run": False,
    }
    write_json(CODEX / "state" / "latest_val41_clean_audit.json", state)
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
