#!/usr/bin/env python3
from pathlib import Path
CODEX = Path('/home/data-gxu/acm/egolink2026-main/code/track2/codex')
runner = CODEX / 'runners' / 'track2_multi_agent_plus.py'
s = runner.read_text(encoding='utf-8')
s = s.replace('''            image_description=image_description + "\n" + str(task_analysis or ""),
''','''            image_description=service_image_description + "\n" + str(service_task_analysis or ""),
''')
s = s.replace('''            "image_description": image_description,
''','''            "image_description": "" if final_compliant else image_description,
''')
s = s.replace('''        if os.environ.get("TRACK2_TEXT_ONLY_VISUAL_CONTEXT", "1") == "1" and image_description:
            service_agent_sys_prompt += "\n\nVideo/action context description from benchmark metadata:\n" + image_description
''','''        if os.environ.get("TRACK2_TEXT_ONLY_VISUAL_CONTEXT", "1") == "1" and service_image_description:
            service_agent_sys_prompt += "\n\nVideo/action context description from benchmark metadata:\n" + service_image_description
''')
s = s.replace('''        if args.scenario == "order" and task_analysis:
            service_agent_sys_prompt += "\n\nOrder layout hint from benchmark dev analysis (dev-only; do not hardcode final answers):\n" + str(task_analysis)
''','''        if args.scenario == "order" and service_task_analysis:
            service_agent_sys_prompt += "\n\nOrder layout hint from benchmark dev analysis (dev-only; do not hardcode final answers):\n" + str(service_task_analysis)
''')
runner.write_text(s, encoding='utf-8')

vs = CODEX / 'scripts' / 'track2_build_visual_state_gpt55.py'
s = vs.read_text(encoding='utf-8')
s = s.replace('''def build_visual_state(scenario: str, number: int, task_index: int, force: bool = False) -> Dict[str, Any]:
    manifest = process_task(scenario, number, task_index, force=False)
''','''def build_visual_state(scenario: str, number: int, task_index: int, force: bool = False) -> Dict[str, Any]:
    if os.environ.get("TRACK2_FINAL_EVAL", "0") == "1":
        state = _empty_state({"scenario": scenario, "frames": []}, "final_mode_no_direct_scenario_json_access")
        state["final_compliant_no_direct_final_json"] = True
        return state
    manifest = process_task(scenario, number, task_index, force=False)
''')
vs.write_text(s, encoding='utf-8')

packer = CODEX / 'scripts' / 'track2_pack_submission.py'
s = packer.read_text(encoding='utf-8')
s = s.replace('''    zip_path = CODEX_ROOT / "submissions" / f"{args.team_name}_track2_{ts}.zip"
''','''    zip_path = CODEX_ROOT / "submissions" / f"{args.team_name}_track2.zip"
    if zip_path.exists() and not args.dry_run:
        backup_zip = CODEX_ROOT / "submissions" / f"{args.team_name}_track2_{ts}.previous.zip"
        zip_path.replace(backup_zip)
''')
s = s.replace("CODEX_ROOT / 'state' / 'best_version.json'", "CODEX_ROOT / 'state' / 'best_track2_api_version.json'")
s = s.replace('''        f"- official_archive_layout: {args.team_name}_track2.zip/{args.team_name}.pdf and results/{args.team_name}/retail6_easy.json retail10_easy.json kitchen4_easy.json restaurant5_easy.json order2_easy.json",
''','''        f"- official_archive_layout: {args.team_name}_track2.zip/{args.team_name}.pdf and results/{args.team_name}/retail6_easy.json retail10_easy.json kitchen4_easy.json restaurant5_easy.json order2_easy.json",
        f"- official_email_subject: {args.team_name}_track2",
''')
packer.write_text(s, encoding='utf-8')

report = CODEX / 'reports' / f'FINAL_COMPLIANCE_PATCH_{__import__("time").strftime("%Y%m%d_%H%M%S")}.md'
report.write_text('''# Final Compliance Patch

- Patched `runners/track2_multi_agent_plus.py` so `--final_eval` / `TRACK2_FINAL_EVAL=1` prevents service-agent prompt, guard state, history metadata, and visual cache from using `image_description` or `analysis/task_analysis` fields from final JSON.
- Simulated user flow still receives official scenario context through the runner, matching the official interaction workflow.
- Patched `track2_build_visual_state_gpt55.py` to return an empty final-compliant state instead of reading final scenario JSON when `TRACK2_FINAL_EVAL=1`.
- Patched `track2_pack_submission.py` to package official layout: `{team_name}_track2.zip` containing `{team_name}.pdf` and `results/{team_name}/retail6_easy.json`, `retail10_easy.json`, `kitchen4_easy.json`, `restaurant5_easy.json`, `order2_easy.json`.
- No final submission was made.
''', encoding='utf-8')
with (CODEX / 'README_STATUS.md').open('a', encoding='utf-8') as f:
    f.write(f'\n## Final Compliance Patch {report.name[-18:-3]}\n\n- Report: `{report}`\n- Service agent final mode no longer receives final JSON `image_description`/`analysis` metadata through the copied runner.\n- Packer uses official `{team_name}_track2.zip` structure.\n- No final submission was made.\n')
print(report)
