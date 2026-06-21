#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import json, os, re, time
from pathlib import Path

EGO = Path('/home/data-gxu/acm/egolink2026-main/code/track2/EgoBench')
CODEX = Path('/home/data-gxu/acm/egolink2026-main/code/track2/codex')
REPORTS = CODEX / 'reports'
REPORTS.mkdir(parents=True, exist_ok=True)

def read(path, max_chars=60000):
    try:
        txt = path.read_text(encoding='utf-8', errors='replace')
    except Exception as e:
        return f'[read error: {e}]'
    return txt[:max_chars]

def grep_lines(text, patterns, limit=40):
    out=[]
    for i,line in enumerate(text.splitlines(),1):
        low=line.lower()
        if any(p.lower() in low for p in patterns):
            out.append((i,line.rstrip()))
            if len(out)>=limit:
                break
    return out

files = {
    'README.md': EGO/'README.md',
    'run/multi_agent.py': EGO/'run'/'multi_agent.py',
    'run/prompts.py': EGO/'run'/'prompts.py',
    'run_all_scenarios.sh': EGO/'run_all_scenarios.sh',
    'analysis_scripts/evaluate_interaction.py': EGO/'analysis_scripts'/'evaluate_interaction.py',
}
texts={k: read(v) for k,v in files.items()}

ts=time.strftime('%Y%m%d_%H%M%S')
report=REPORTS/f'V12_OFFICIAL_BASELINE_AUDIT_{ts}.md'

# Extract compact evidence snippets.
readme_hits=grep_lines(texts['README.md'], ['final_eval','tool','json','submission','image','video','service agent','result'], 80)
runner_hits=grep_lines(texts['run/multi_agent.py'], ['SERVICE_AGENT_PROMPT_BASE','image_description','build_message_with_image','check_tool_call','execute','max_turn','max_tool','summary','user_instruction','history'], 100)
prompt_hits=grep_lines(texts['run/prompts.py'], ['SERVICE_AGENT_PROMPT_BASE','tool','json','image','video','professional service agent','workflow','parallel','natural'], 100)
eval_hits=grep_lines(texts['analysis_scripts/evaluate_interaction.py'], ['result_success','tool_success','joint','micro','fuzzy','db hash','compare','matched','ground_truth','tool call'], 120)

md=[]
md.append(f'# V12 Official Baseline Audit\n')
md.append(f'- generated_at: `{ts}`')
md.append(f'- EgoBench root: `{EGO}`')
md.append(f'- codex root: `{CODEX}`')
md.append('- source mode: read-only audit; official source files were not modified')
md.append('')
md.append('## Official Agent Input Surface')
md.append('- The official runner loads scenario JSON and tool definitions, initializes a per-task DB, then runs a two-agent loop: simulated user plus service agent.')
md.append('- The service agent receives the official service prompt with full tool descriptions, the dialogue history, and visual/video context through `build_message_with_image(... use_vision=True ...)` for the first service user message.')
md.append('- The simulated user receives `user_instruction` and `image_description`; for final compliance, our service wrapper must not directly consume hidden final JSON metadata beyond the official interaction surface.')
md.append('- Service history starts empty and accumulates user messages, assistant tool calls, tool observations, and final natural-language responses.')
md.append('')
md.append('## Official Prompt / Tool Output Style')
md.append('- Official prompt is compact and general: act as a professional service agent, understand context including image/video, call tools only when needed, and answer concisely otherwise.')
md.append('- Tool-call turns must be a pure JSON array, for example `[ {"tool_name": "...", "parameters": {...}} ]`; natural-language turns must not include JSON tool-call text.')
md.append('- Independent retrieval calls may be parallelized in a single JSON array; dependent calls must wait for tool observations.')
md.append('- The runner checks whether the service text is a tool call, executes tools, appends observations, and continues the same user turn until the service emits a natural reply.')
md.append('')
md.append('## Official Tool Loop')
md.append('- Max conversation turns are bounded by the official runner; tool calls also have a safety cap.')
md.append('- Tool outputs are part of the service-agent history, so process coverage is naturally induced by the loop rather than by a separate hard FSM.')
md.append('- This matters for V12: imitate the official loop/prompt shape, then add memory/guard/visual cards as concise adjuncts instead of replacing the baseline style with a long rigid planner.')
md.append('')
md.append('## Evaluator Logic')
md.append('- `result_success` is based on replaying predicted and ground-truth tool chains from the same initial DB and comparing final DB state/hash.')
md.append('- `tool_success` compares the predicted tool trajectory against ground truth after filtering parameters to DB method signatures; fuzzy matching applies to entity-name keys by scenario.')
md.append('- `joint_success` requires both result and tool success. Micro accuracy counts matched GT tool calls over total GT tool calls and is diagnostic, not a substitute for joint success.')
md.append('- Scenario fuzzy keys include retail `product_name`, kitchen `ingredient_name`/`recipe_name`, restaurant `dish_name`/`set_meal_name`, and order `dish_name`/`set_meal_name`/`restaurant_name`.')
md.append('')
md.append('## Why Official Video-MLLM Baseline Style Can Help')
md.append('- The official baseline keeps visual context in the actual service-agent message rather than only as a text prior. This can reduce visual follow-up loops and wrong entity grounding in retail/order/restaurant tasks.')
md.append('- Its prompt is short and tool-schema-native, which tends to preserve valid JSON tool trajectories and avoids diluting the model with excessive rules.')
md.append('- Its tool loop lets the model see observations immediately and continue within the same user turn, matching evaluator process expectations.')
md.append('- V12 should therefore keep the official interaction style as the top prompt frame, then inject V10 memory, scoring checklist, Qwen3-VL grounding, and soft guard signals as compact evidence cards.')
md.append('')
md.append('## V12 Implementation Implications')
md.append('- Do not modify official EgoBench source.')
md.append('- Use copied runner/PYTHONPATH wrapper and preserve official JSON-array/natural-language surface.')
md.append('- Add Qwen3-VL as visual grounding teacher only. It must not call tools or directly decide final tool actions.')
md.append('- Keep V10 innovations enabled as soft evidence: evaluator awareness, memory retrieval, visual resolver, retail trimming, order synthesis, guard/rerank signals.')
md.append('- Do not run final in this phase; only run 4-gate: retail9, restaurant4, order1, kitchen2.')
md.append('')
for title,hits in [('README Evidence',readme_hits),('Runner Evidence',runner_hits),('Prompt Evidence',prompt_hits),('Evaluator Evidence',eval_hits)]:
    md.append(f'## {title}')
    if not hits:
        md.append('- no matching snippets captured')
    else:
        for line_no,line in hits[:60]:
            md.append(f'- `{line_no}`: `{line[:220]}`')
    md.append('')

report.write_text('\n'.join(md), encoding='utf-8')
print(report)
