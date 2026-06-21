#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import textwrap
import time
import zipfile
from pathlib import Path
from typing import Any


CODEX = Path(os.environ.get("CODEX_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/codex"))
EGO = Path(os.environ.get("EGO_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/EgoBench"))
TEAM = "V10_full_memory_final_candidate_draft"
RUN_ID = "V10_full_memory_final_candidate_draft_20260618_1940"
EXPECTED = {
    "retail6_easy.json": 49,
    "retail10_easy.json": 63,
    "kitchen4_easy.json": 50,
    "restaurant5_easy.json": 50,
    "order2_easy.json": 97,
}


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def latest_report(prefix: str) -> Path | None:
    files = sorted((CODEX / "reports").glob(prefix + "*.md"), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def result_stats() -> dict[str, Any]:
    root = EGO / "results" / TEAM
    stats: dict[str, Any] = {
        "team": TEAM,
        "root": str(root),
        "files": {},
        "missing_files": [],
        "total_tasks": 0,
        "empty_dialogue": 0,
        "empty_tool_calls": 0,
        "api_error_hits": 0,
        "timeout_hits": 0,
        "mixed_json_text_hits": 0,
        "final_hidden_metadata_leaks": 0,
        "total_turns": 0,
        "total_tool_calls": 0,
        "max_tool_calls": 0,
        "result_file_sha256": {},
    }
    for name, expected in EXPECTED.items():
        path = root / name
        if not path.exists():
            stats["missing_files"].append(name)
            stats["files"][name] = {"exists": False, "expected": expected}
            continue
        stats["result_file_sha256"][name] = sha256(path)
        data = read_json(path)
        if not isinstance(data, list):
            stats["files"][name] = {"exists": True, "valid_json_list": False, "expected": expected}
            continue
        turns = []
        tool_counts = []
        for item in data:
            if not isinstance(item, dict):
                continue
            stats["total_tasks"] += 1
            if item.get("instruction") or item.get("image_description") or item.get("analysis"):
                stats["final_hidden_metadata_leaks"] += 1
            dialogue = item.get("dialogue") or []
            tool_calls = item.get("tool_calls") or []
            if not dialogue:
                stats["empty_dialogue"] += 1
            if not tool_calls:
                stats["empty_tool_calls"] += 1
            text = json.dumps(item, ensure_ascii=False).lower()
            if any(s in text for s in ("api_error", "readtimeout", "connectionerror", "rate limit", "429")):
                stats["api_error_hits"] += 1
            if "timeout" in text or "timed out" in text:
                stats["timeout_hits"] += 1
            if "```json" in text and "tool_name" in text:
                stats["mixed_json_text_hits"] += 1
            rounds = item.get("rounds_count", len([d for d in dialogue if isinstance(d, dict) and d.get("role") == "user"]))
            calls = item.get("tool_calls_count", sum(len(t.get("calls") or []) for t in tool_calls if isinstance(t, dict)))
            if isinstance(rounds, (int, float)):
                turns.append(rounds)
            if isinstance(calls, (int, float)):
                tool_counts.append(calls)
        stats["files"][name] = {
            "exists": True,
            "valid_json_list": True,
            "tasks": len(data),
            "expected": expected,
            "count_ok": len(data) == expected,
        }
        stats["total_turns"] += sum(turns)
        stats["total_tool_calls"] += sum(tool_counts)
        stats["max_tool_calls"] = max([stats["max_tool_calls"]] + [int(x) for x in tool_counts])
    stats["task_count_ok"] = stats["total_tasks"] == sum(EXPECTED.values())
    stats["all_files_ok"] = (
        not stats["missing_files"]
        and all(info.get("count_ok") for info in stats["files"].values())
    )
    stats["avg_turns"] = stats["total_turns"] / stats["total_tasks"] if stats["total_tasks"] else 0.0
    stats["avg_tool_calls"] = stats["total_tool_calls"] / stats["total_tasks"] if stats["total_tasks"] else 0.0
    return stats


def memory_stats() -> dict[str, Any]:
    base = CODEX / "runs" / "V10_full_memory_final_candidate_draft"
    out = {
        "memory_hit_records": 0,
        "selected_cards_total": 0,
        "no_final_metadata_false": 0,
        "run_dirs": [],
    }
    if not base.exists():
        return out
    for run_dir in sorted(base.iterdir()):
        hit_dir = run_dir / "memory_hits"
        if not hit_dir.exists():
            continue
        out["run_dirs"].append(run_dir.name)
        for path in hit_dir.glob("*.jsonl"):
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                out["memory_hit_records"] += 1
                out["selected_cards_total"] += len(obj.get("selected_card_ids") or [])
                if not obj.get("no_final_metadata", False):
                    out["no_final_metadata_false"] += 1
    return out


def report_lines(stats: dict[str, Any], mem: dict[str, Any]) -> list[str]:
    manifest = read_json(CODEX / "memory_bank_v10" / "memory_bank_manifest.json", {})
    coverage = read_json(CODEX / "memory_bank_v10" / "memory_coverage_audit.json", {})
    lines = [
        "# V10 Final Full Sanity",
        "",
        f"- timestamp: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        f"- team_name: `{TEAM}`",
        f"- official_final_tasks_expected: {sum(EXPECTED.values())}",
        f"- total_tasks: {stats['total_tasks']}",
        f"- task_count_ok_309: {stats['task_count_ok']}",
        f"- all_files_ok: {stats['all_files_ok']}",
        f"- missing_files: {stats['missing_files']}",
        f"- final_hidden_metadata_leaks: {stats['final_hidden_metadata_leaks']}",
        f"- empty_dialogue: {stats['empty_dialogue']}",
        f"- empty_tool_calls: {stats['empty_tool_calls']}",
        f"- api_error_hits: {stats['api_error_hits']}",
        f"- timeout_hits: {stats['timeout_hits']}",
        f"- mixed_json_text_hits: {stats['mixed_json_text_hits']}",
        f"- avg_turns: {stats['avg_turns']:.2f}",
        f"- avg_tool_calls: {stats['avg_tool_calls']:.2f}",
        f"- max_tool_calls: {stats['max_tool_calls']}",
        f"- memory_hit_records: {mem['memory_hit_records']}",
        f"- memory_selected_cards_total: {mem['selected_cards_total']}",
        f"- memory_no_final_metadata_false: {mem['no_final_metadata_false']}",
        f"- memory_bank_manifest: `{CODEX / 'memory_bank_v10' / 'memory_bank_manifest.json'}`",
        f"- dev_offline_tasks_in_memory_bank: {manifest.get('tasks_with_gt', manifest.get('tasks_entered_memory', coverage.get('tasks_entered_memory', 'unknown')))}",
        "- auto_submitted: no",
        "",
        "## Per File",
    ]
    for name, info in stats["files"].items():
        lines.append(f"- {name}: {info}")
    lines += [
        "",
        "## Prior Validation",
        "",
        "- V10 A_medium sanity: joint 12.20%, result 17.07%, tool 12.20%, micro 29.49%.",
        "- V9_5 A_medium baseline: joint 12.20%, micro 26.28%.",
        "- V10 final smoke: passed before full final run.",
        "",
        "## Compliance",
        "",
        "- Final run uses `--final_eval`.",
        "- Service-side prompt receives no final JSON `Instruction`, `image_description`, or `analysis` fields.",
        "- Memory bank excludes official final scenarios and is marked `no_final_metadata` in retrieval telemetry.",
        "- GPT-5.5 is used as the main service agent through an OpenAI-compatible endpoint.",
        "- DeepSeek crosscheck was disabled because no DeepSeek key was available in this run.",
        "- No automatic submission was performed.",
    ]
    return lines


def write_text(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def escape_pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def write_simple_pdf(path: Path, title: str, body_lines: list[str]) -> None:
    width, height = 612, 792
    margin_x, margin_y = 48, 56
    line_height = 13
    max_chars = 88
    wrapped: list[str] = [title, ""]
    for line in body_lines:
        plain = re.sub(r"`([^`]*)`", r"\1", line)
        plain = plain.replace("# ", "").replace("## ", "")
        if not plain:
            wrapped.append("")
            continue
        wrapped.extend(textwrap.wrap(plain, width=max_chars) or [""])
    lines_per_page = max(1, math.floor((height - margin_y * 2) / line_height))
    pages = [wrapped[i : i + lines_per_page] for i in range(0, len(wrapped), lines_per_page)] or [[]]
    objects: list[bytes] = []
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    kids = " ".join(f"{3 + i * 2} 0 R" for i in range(len(pages)))
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode("ascii"))
    font_obj_num = 3 + len(pages) * 2
    for idx, page_lines in enumerate(pages):
        page_obj = 3 + idx * 2
        content_obj = page_obj + 1
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width} {height}] "
            f"/Resources << /Font << /F1 {font_obj_num} 0 R >> >> /Contents {content_obj} 0 R >>".encode("ascii")
        )
        commands = ["BT", "/F1 10 Tf", f"{margin_x} {height - margin_y} Td"]
        for line_no, line in enumerate(page_lines):
            if line_no:
                commands.append(f"0 -{line_height} Td")
            commands.append(f"({escape_pdf_text(line)}) Tj")
        commands.append("ET")
        stream = "\n".join(commands).encode("latin-1", errors="replace")
        objects.append(b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for idx, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out.extend(f"{idx} 0 obj\n".encode("ascii"))
        out.extend(obj)
        out.extend(b"\nendobj\n")
    xref_offset = len(out)
    out.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    out.extend(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        out.extend(f"{off:010d} 00000 n \n".encode("ascii"))
    out.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode("ascii")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(out))


def create_zip(pdf_path: Path, stats: dict[str, Any]) -> Path:
    zip_path = CODEX / "submissions" / f"{TEAM}_track2.zip"
    if zip_path.exists():
        backup = CODEX / "submissions" / f"{TEAM}_track2_{time.strftime('%Y%m%d_%H%M%S')}.previous.zip"
        zip_path.replace(backup)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    result_root = EGO / "results" / TEAM
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(pdf_path, arcname=f"{TEAM}.pdf")
        for name in EXPECTED:
            path = result_root / name
            if path.exists():
                zf.write(path, arcname=f"results/{TEAM}/{name}")
    return zip_path


def main() -> None:
    ts = time.strftime("%Y%m%d_%H%M%S")
    stats = result_stats()
    mem = memory_stats()
    lines = report_lines(stats, mem)
    full_report = CODEX / "reports" / f"V10_FINAL_FULL_SANITY_{ts}.md"
    write_text(full_report, lines)

    tech_md = CODEX / "reports" / f"V10_TECHNICAL_REPORT_DRAFT_{RUN_ID}.md"
    tech_lines = [
        "# EgoLink Track2 V10 Technical Report Draft",
        "",
        "This draft accompanies the non-submitted Track2 final candidate package.",
        "",
        "## Method Summary",
        "",
        "- Main service model: GPT-5.5 through an OpenAI-compatible endpoint.",
        "- Memory/rule bank: dev/offline-only tool constitution, process templates, scoring cards, failure patterns, success abstractions, and canonicalization cards.",
        "- Guards: schema repair, duplicate mutation guard, pinned user/restaurant identifiers, retail narrowing, order process synthesis, evaluator-aware checklist, and deterministic rerank signals.",
        "- Final compliance: no direct service-agent access to final hidden JSON metadata; no final answer hardcoding; no auto-submission.",
        "",
        "## Sanity Summary",
        "",
        *lines[2:],
    ]
    write_text(tech_md, tech_lines)
    pdf_path = CODEX / "reports" / f"{TEAM}.pdf"
    write_simple_pdf(pdf_path, "EgoLink Track2 V10 Technical Report Draft", tech_lines)

    zip_path = create_zip(pdf_path, stats)
    zip_hash = sha256(zip_path)
    package_report = CODEX / "reports" / f"V10_SUBMISSION_PACKAGE_DRAFT_{ts}.md"
    package_lines = [
        "# V10 Submission Package Draft",
        "",
        f"- timestamp: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        f"- team_name: `{TEAM}`",
        f"- zip_path: `{zip_path}`",
        f"- zip_sha256: `{zip_hash}`",
        f"- technical_report_pdf: `{pdf_path}`",
        f"- final_full_sanity_report: `{full_report}`",
        f"- auto_submitted: no",
        "- official layout: yes, if all five result files are present.",
        "",
        "## Zip Contents",
    ]
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            package_lines.append(f"- {info.filename} ({info.file_size} bytes)")
    write_text(package_report, package_lines)

    readme = CODEX / "reports" / f"V10_FINAL_CANDIDATE_README_{ts}.md"
    compliance_risk = not (
        stats["task_count_ok"]
        and stats["all_files_ok"]
        and stats["final_hidden_metadata_leaks"] == 0
        and mem["no_final_metadata_false"] == 0
    )
    readme_lines = [
        "# V10 Final Candidate README",
        "",
        f"- memory bank dev/offline task count: see `{CODEX / 'reports' / 'V10_FULL_MEMORY_BANK_BUILD_20260618_171530.md'}`; 649 tasks entered memory in the completed build.",
        "- V10 A_medium vs V9_5: joint equal at 12.20%; micro improved from 26.28% to 29.49%.",
        "- final smoke: passed.",
        f"- final 309 completed: {stats['task_count_ok']}",
        f"- submission zip: `{zip_path}`",
        f"- compliance risk detected by sanity checks: {compliance_risk}",
        "- recommendation: manual review before any submission; this script did not submit anything.",
        "",
        f"- final sanity: `{full_report}`",
        f"- package report: `{package_report}`",
        f"- technical report pdf: `{pdf_path}`",
    ]
    write_text(readme, readme_lines)
    state = {
        "team": TEAM,
        "run_id": RUN_ID,
        "final_full_sanity_report": str(full_report),
        "technical_report_md": str(tech_md),
        "technical_report_pdf": str(pdf_path),
        "package_report": str(package_report),
        "candidate_readme": str(readme),
        "zip_path": str(zip_path),
        "zip_sha256": zip_hash,
        "stats": stats,
        "memory": mem,
        "auto_submitted": False,
    }
    (CODEX / "state" / "v10_final_candidate_package.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    with (CODEX / "README_STATUS.md").open("a", encoding="utf-8") as f:
        f.write(
            "\n## V10 Final Candidate Draft Package "
            + ts
            + "\n\n"
            + f"- final_full_sanity: `{full_report}`\n"
            + f"- submission_zip: `{zip_path}`\n"
            + f"- auto_submitted: no\n"
        )
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
