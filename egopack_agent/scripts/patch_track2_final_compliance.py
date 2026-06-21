#!/usr/bin/env python3
from pathlib import Path
p=Path('/home/data-gxu/acm/egolink2026-main/code/track2/codex/runners/track2_multi_agent_plus.py')
s=p.read_text(encoding='utf-8')
s=s.replace('''        user_instruction = sc.get("Instruction", "")
        image_path = sc.get("image_path", None)
        image_path = get_video_url_for_model(image_path, args.service_model_name)
        image_description = sc.get("image_description", "")
        task_analysis = sc.get("analysis", sc.get("Analysis", sc.get("task_analysis", "")))
        visual_context = load_v6_visual_context(args.scenario, args.scenario_number, task_id)
''','''        user_instruction = sc.get("Instruction", "")
        image_path = sc.get("image_path", None)
        image_path = get_video_url_for_model(image_path, args.service_model_name)
        image_description = sc.get("image_description", "")
        task_analysis = sc.get("analysis", sc.get("Analysis", sc.get("task_analysis", "")))
        final_compliant = bool(getattr(args, "final_eval", False) or os.environ.get("TRACK2_FINAL_EVAL", "0") == "1")
        # Official final-stage rule: the service agent must not directly use
        # scenarios/final JSON metadata. The simulated user may receive it via
        # the official runner flow, but service-side prompts/guards cannot.
        service_image_description = "" if final_compliant else image_description
        service_task_analysis = "" if final_compliant else task_analysis
        visual_context = {} if final_compliant else load_v6_visual_context(args.scenario, args.scenario_number, task_id)
''')
s=s.replace('''            image_description=image_description + "\n" + str(task_analysis or ""),
''','''            image_description=service_image_description + "\n" + str(service_task_analysis or ""),
''')
s=s.replace('''            "visual_cache_id": visual_context.get("cache_id"),
            "visual_state_present": bool(visual_context.get("visual_state_text")),
            "contact_sheet_path": visual_context.get("contact_sheet"),
            "analysis": task_analysis,
''','''            "visual_cache_id": visual_context.get("cache_id"),
            "visual_state_present": bool(visual_context.get("visual_state_text")),
            "contact_sheet_path": visual_context.get("contact_sheet"),
            "analysis": "" if final_compliant else task_analysis,
            "final_compliant_no_direct_final_json": final_compliant,
''')
s=s.replace('''        if os.environ.get("TRACK2_TEXT_ONLY_VISUAL_CONTEXT", "1") == "1" and image_description:
            service_agent_sys_prompt += "\n\nVideo/action context description from benchmark metadata:\n" + image_description
        if os.environ.get("TRACK2_ENABLE_VISUAL_CACHE", "0") == "1" and visual_context.get("visual_state_text"):
            service_agent_sys_prompt += "\n\nCached visual_state evidence:\n" + visual_context["visual_state_text"]
        if args.scenario == "order" and task_analysis:
            service_agent_sys_prompt += "\n\nOrder layout hint from benchmark dev analysis (dev-only; do not hardcode final answers):\n" + str(task_analysis)
''','''        if os.environ.get("TRACK2_TEXT_ONLY_VISUAL_CONTEXT", "1") == "1" and service_image_description:
            service_agent_sys_prompt += "\n\nVideo/action context description from benchmark metadata:\n" + service_image_description
        if os.environ.get("TRACK2_ENABLE_VISUAL_CACHE", "0") == "1" and visual_context.get("visual_state_text"):
            service_agent_sys_prompt += "\n\nCached visual_state evidence:\n" + visual_context["visual_state_text"]
        if args.scenario == "order" and service_task_analysis:
            service_agent_sys_prompt += "\n\nOrder layout hint from benchmark dev analysis (dev-only; do not hardcode final answers):\n" + str(service_task_analysis)
''')
s=s.replace('''    parser.add_argument(
        "--num_tasks",
        type=int,
        default=0,
        help="Number of tasks to test from the beginning of the scenario. 0 means test all tasks."
    )

    args = parser.parse_args()
''','''    parser.add_argument(
        "--num_tasks",
        type=int,
        default=0,
        help="Number of tasks to test from the beginning of the scenario. 0 means test all tasks."
    )

    parser.add_argument(
        "--final_eval",
        action="store_true",
        help="Run in official final-compliant mode: do not expose final JSON metadata to the service agent."
    )

    args = parser.parse_args()
    if args.final_eval:
        os.environ["TRACK2_FINAL_EVAL"] = "1"
''')
p.write_text(s,encoding='utf-8')

p=Path('/home/data-gxu/acm/egolink2026-main/code/track2/codex/scripts/track2_build_visual_state_gpt55.py')
s=p.read_text(encoding='utf-8')
s=s.replace('''def _task_context(scenario: str, number: int, task_index: int) -> Dict[str, Any]:
    path = EGO_ROOT / "scenarios" / "final" / f"{scenario}{number}.json"
    tasks = json.loads(path.read_text(encoding="utf-8"))
    task = tasks[max(0, task_index - 1)]
    return {
        "Instruction": task.get("Instruction", ""),
        "image_description": task.get("image_description", ""),
        "analysis": task.get("analysis", task.get("Analysis", task.get("task_analysis", ""))),
        "image_path": task.get("image_path", ""),
    }
''','''def _task_context(scenario: str, number: int, task_index: int) -> Dict[str, Any]:
    if os.environ.get("TRACK2_FINAL_EVAL", "0") == "1":
        return {
            "scenario": f"{scenario}{number}",
            "task_index": task_index,
            "final_compliant": True,
            "note": "Final mode: scenarios/final JSON metadata is not read for service-agent visual_state.",
        }
    path = EGO_ROOT / "scenarios" / "final" / f"{scenario}{number}.json"
    tasks = json.loads(path.read_text(encoding="utf-8"))
    task = tasks[max(0, task_index - 1)]
    return {
        "Instruction": task.get("Instruction", ""),
        "image_description": task.get("image_description", ""),
        "analysis": task.get("analysis", task.get("Analysis", task.get("task_analysis", ""))),
        "image_path": task.get("image_path", ""),
    }
''')
p.write_text(s,encoding='utf-8')

p=Path('/home/data-gxu/acm/egolink2026-main/code/track2/codex/scripts/track2_pack_submission.py')
s=p.read_text(encoding='utf-8')
s=s.replace('''    parser.add_argument("--model-name", default=os.environ.get("SERVICE_MODEL_NAME", "deepseek-v4-pro"))
''','''    parser.add_argument("--model-name", default=os.environ.get("SERVICE_MODEL_NAME", "deepseek-v4-pro"))
    parser.add_argument("--team-name", default=os.environ.get("TRACK2_TEAM_NAME", "egolink_codex_track2"))
    parser.add_argument("--technical-report", default="", help="Path to {team_name}.pdf; package notes if absent.")
''')
s=s.replace('''    result_root = EGO_ROOT / "results" / args.model_name
    files = sorted(result_root.glob("*.json")) if result_root.exists() else []
    zip_path = CODEX_ROOT / "submissions" / f"track2_final_{args.model_name}_{ts}.zip"
    readme = CODEX_ROOT / "reports" / f"FINAL_SUBMISSION_README_{ts}.md"
    if not args.dry_run:
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for f in files:
                zf.write(f, arcname=f"submission/results/{args.model_name}/{f.name}")
    missing = []
    for name in ["retail6_easy.json", "retail10_easy.json", "kitchen4_easy.json", "restaurant5_easy.json", "order2_easy.json"]:
        if not (result_root / name).exists():
            missing.append(name)
''','''    result_root = EGO_ROOT / "results" / args.model_name
    required = ["retail6_easy.json", "retail10_easy.json", "kitchen4_easy.json", "restaurant5_easy.json", "order2_easy.json"]
    files = [result_root / name for name in required if (result_root / name).exists()]
    missing = [name for name in required if not (result_root / name).exists()]
    zip_path = CODEX_ROOT / "submissions" / f"{args.team_name}_track2_{ts}.zip"
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
''')
s=s.replace('''        f"- model_name: {args.model_name}",
''','''        f"- model_name: {args.model_name}",
        f"- team_name: {args.team_name}",
''')
s=s.replace('''        f"- final_task_count_files: {len(files)}",
''','''        f"- final_required_files_present: {len(files)}/5",
        f"- technical_report_pdf_present: {not report_missing}",
''')
s=s.replace('''        f"- missing_tasks: {missing}",
''','''        f"- missing_result_files: {missing}",
        f"- official_archive_layout: {args.team_name}_track2.zip/{args.team_name}.pdf and results/{args.team_name}/retail6_easy.json retail10_easy.json kitchen4_easy.json restaurant5_easy.json order2_easy.json",
''')
p.write_text(s,encoding='utf-8')

# Correct the just-written guide/report wording.
for rel in ['reports/FINAL_STAGE_SUBMISSION_GUIDE_20260617_170840.md','reports/OFFICIAL_FINAL_MIN_SYNC_20260617_170840.md']:
    p=Path('/home/data-gxu/acm/egolink2026-main/code/track2/codex')/rel
    if p.exists():
        s=p.read_text(encoding='utf-8')
        s=s.replace('`retail_easy.json`, `kitchen_easy.json`, `restaurant_easy.json`, `order_easy.json`','`retail6_easy.json`, `retail10_easy.json`, `kitchen4_easy.json`, `restaurant5_easy.json`, `order2_easy.json`')
        s=s.replace('contains final output names: `False`','contains official final output names: `True`')
        p.write_text(s,encoding='utf-8')
