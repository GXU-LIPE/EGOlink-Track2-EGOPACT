#!/usr/bin/env python3
from pathlib import Path
import time
CODEX=Path('/home/data-gxu/acm/egolink2026-main/code/track2/codex')
p=CODEX/'runners'/'track2_multi_agent_plus.py'
s=p.read_text(encoding='utf-8')
s=s.replace('image_description=image_description + "\\n" + str(task_analysis or ""),','image_description=service_image_description + "\\n" + str(service_task_analysis or ""),')
s=s.replace('if os.environ.get("TRACK2_TEXT_ONLY_VISUAL_CONTEXT", "1") == "1" and image_description:\n            service_agent_sys_prompt += "\\n\\nVideo/action context description from benchmark metadata:\\n" + image_description','if os.environ.get("TRACK2_TEXT_ONLY_VISUAL_CONTEXT", "1") == "1" and service_image_description:\n            service_agent_sys_prompt += "\\n\\nVideo/action context description from benchmark metadata:\\n" + service_image_description')
s=s.replace('if args.scenario == "order" and task_analysis:\n            service_agent_sys_prompt += "\\n\\nOrder layout hint from benchmark dev analysis (dev-only; do not hardcode final answers):\\n" + str(task_analysis)','if args.scenario == "order" and service_task_analysis:\n            service_agent_sys_prompt += "\\n\\nOrder layout hint from benchmark dev analysis (dev-only; do not hardcode final answers):\\n" + str(service_task_analysis)')
p.write_text(s,encoding='utf-8')
ts=time.strftime('%Y%m%d_%H%M%S')
report=CODEX/'reports'/f'FINAL_COMPLIANCE_PATCH_{ts}.md'
report.write_text('''# Final Compliance Patch

- Completed service-side final compliance cleanup in `runners/track2_multi_agent_plus.py`.
- In `--final_eval` / `TRACK2_FINAL_EVAL=1`, service prompt, episode guard state, visual cache injection, and order layout hint no longer receive final JSON `image_description` or `analysis/task_analysis` metadata.
- Simulated user prompts still receive scenario context through the official runner flow, which matches the README Q&A.
- `track2_build_visual_state_gpt55.py` returns an empty final-compliant state instead of reading final JSON when final mode is active.
- `track2_pack_submission.py` now targets the official zip layout: `{team_name}_track2.zip` with `{team_name}.pdf` and `results/{team_name}/retail6_easy.json`, `retail10_easy.json`, `kitchen4_easy.json`, `restaurant5_easy.json`, `order2_easy.json`.
- No final submission was made.
''',encoding='utf-8')
with (CODEX/'README_STATUS.md').open('a',encoding='utf-8') as f:
    f.write(f'\n## Final Compliance Patch {ts}\n\n- Report: `{report}`\n- Service agent final mode no longer receives final JSON `image_description`/`analysis` metadata.\n- Packer uses official `{team_name}_track2.zip` structure.\n- No final submission was made.\n'.replace('{team_name}','{team_name}'))
print(report)
