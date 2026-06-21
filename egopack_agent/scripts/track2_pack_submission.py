# -*- coding: utf-8 -*-
"""Package Track2 final results without submitting."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import zipfile


EGO_ROOT = Path(os.environ.get("EGO_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/EgoBench"))
CODEX_ROOT = Path(os.environ.get("CODEX_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/codex"))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="")
    parser.add_argument("--model-name", default=os.environ.get("SERVICE_MODEL_NAME", "deepseek-v4-pro"))
    parser.add_argument("--team-name", default=os.environ.get("TRACK2_TEAM_NAME", "egolink_codex_track2"))
    parser.add_argument("--technical-report", default="", help="Path to {team_name}.pdf; package notes if absent.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    ts = time.strftime("%Y%m%d_%H%M%S")
    result_root = EGO_ROOT / "results" / args.model_name
    required = ["retail6_easy.json", "retail10_easy.json", "kitchen4_easy.json", "restaurant5_easy.json", "order2_easy.json"]
    files = [result_root / name for name in required if (result_root / name).exists()]
    missing = [name for name in required if not (result_root / name).exists()]
    zip_path = CODEX_ROOT / "submissions" / f"{args.team_name}_track2.zip"
    if zip_path.exists() and not args.dry_run:
        backup_zip = CODEX_ROOT / "submissions" / f"{args.team_name}_track2_{ts}.previous.zip"
        zip_path.replace(backup_zip)
    readme = CODEX_ROOT / "reports" / f"FINAL_SUBMISSION_README_{ts}.md"
    report_path = Path(args.technical_report) if args.technical_report else None
    report_missing = not (report_path and report_path.exists())
    if not args.dry_run:
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for f in files:
                zf.write(f, arcname=f"results/{args.team_name}/{f.name}")
            if report_path and report_path.exists():
                zf.write(report_path, arcname=f"{args.team_name}.pdf")
    lines = [
        "# Track2 Final Submission Package",
        "",
        f"- timestamp: {ts}",
        f"- model_name: {args.model_name}",
        f"- team_name: {args.team_name}",
        f"- ego_root: {EGO_ROOT}",
        f"- wrapper_version: {json.dumps(json.load(open(CODEX_ROOT / 'state' / 'best_track2_api_version.json', encoding='utf-8')), ensure_ascii=False) if (CODEX_ROOT / 'state' / 'best_track2_api_version.json').exists() else 'unknown'}",
        f"- final_required_files_present: {len(files)}/5",
        f"- technical_report_pdf_present: {not report_missing}",
        f"- zip: {zip_path}",
        f"- zip_sha256: {sha256(zip_path) if zip_path.exists() else ''}",
        f"- missing_result_files: {missing}",
        f"- official_archive_layout: {args.team_name}_track2.zip/{args.team_name}.pdf and results/{args.team_name}/retail6_easy.json retail10_easy.json kitchen4_easy.json restaurant5_easy.json order2_easy.json",
        f"- official_email_subject: {args.team_name}_track2",
        f"- abnormal_tasks: []",
        "- auto_submitted: false",
        "",
        "Result files:",
    ]
    for f in files:
        lines.append(f"- {f.name} sha256={sha256(f)}")
    if not args.dry_run:
        readme.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
