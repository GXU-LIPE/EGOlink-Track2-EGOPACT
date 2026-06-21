#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, pathlib, time, re
CODEX=pathlib.Path('/home/data-gxu/acm/egolink2026-main/code/track2/codex')
RUN='V9_5_memory_deepseek_rerank_validation_A_medium_20260618_1420_frozen_noproxy'
VERSION='V9_5_memory_deepseek_rerank'
BASE=CODEX/'runs'/VERSION/RUN
summary=json.loads((BASE/'eval_summary.json').read_text()).get('summary',{})
rows=json.loads((BASE/'eval_summary.json').read_text()).get('rows',[])
prior={
 'V9_4_5_visual_retrieval_fix': CODEX/'runs'/'V9_4_5_visual_retrieval_fix'/'V9_4_5_visual_retrieval_fix_validation_A_medium_20260618_1014'/'eval_summary.json',
 'V9_5_old_noproxy': CODEX/'runs'/'V9_5_memory_deepseek_rerank'/'V9_5_memory_deepseek_rerank_validation_A_medium_20260618_1218_noproxy'/'eval_summary.json',
}
comp=[]
for name,path in prior.items():
    if path.exists():
        s=json.loads(path.read_text()).get('summary',{})
        comp.append((name,s))
comp.append(('V9_5_frozen_noproxy', summary))
# telemetry counts from wrapper/v8 events
events={}
for folder in [BASE/'wrapper_events', BASE/'v8_events']:
    if not folder.exists(): continue
    for f in folder.glob('*.jsonl'):
        for line in f.read_text(errors='ignore').splitlines():
            try: obj=json.loads(line)
            except Exception: continue
            ev=str(obj.get('event') or obj.get('event_name') or '')
            if ev:
                events[ev]=events.get(ev,0)+1
# report
now=time.strftime('%Y%m%d_%H%M%S')
report=CODEX/'reports'/f'V9_5_FROZEN_A_MEDIUM_ANALYSIS_{now}.md'
lines=[f'# V9.5 Frozen A Medium Analysis {now}','',f'- run_id: `{RUN}`','- version: `V9_5_memory_deepseek_rerank`','- split: `state/materialized_splits/validation_A_limit30` frozen from V9_4_5 41-task subset','- no_proxy_api: true','- deepseek_online_crosscheck: false','- protected_best_updated: false','- final_submission: not submitted','', '## Headline','', '- Split hygiene is fixed: validation_A_medium stayed at 41 valid tasks and retained `kitchen2::33`.','- Reranker scoring bug was fixed by parsing JSON-string candidates before process/counterfactual scoring.','- New V9.5 recovered V9_4_5 joint/tool and slightly improved micro, but it is still far below protected V6. Do not run holdout/final from this candidate.','', '## Comparison','', '| version | valid | joint | result | tool | micro | calls | note |','|---|---:|---:|---:|---:|---:|---:|---|']
for name,s in comp:
    note='same frozen 41-task split' if name in {'V9_4_5_visual_retrieval_fix','V9_5_frozen_noproxy'} else 'old run had split drift, valid=40'
    lines.append(f"| {name} | {s.get('valid',0)} | {s.get('joint',0):.4f} | {s.get('result',0):.4f} | {s.get('tool',0):.4f} | {s.get('micro',0):.4f} | {s.get('correct_calls',0)}/{s.get('gt_calls',0)} | {note} |")
lines += ['', '## Per Scenario File Metrics', '', '| scenario | valid | joint | result | tool | micro | calls |','|---|---:|---:|---:|---:|---:|---:|']
for r in rows:
    lines.append(f"| {r.get('scenario')}{r.get('number')} | {r.get('valid',0)} | {r.get('joint',0):.3f} | {r.get('result',0):.3f} | {r.get('tool',0):.3f} | {r.get('micro',0):.3f} | {r.get('correct_calls',0)}/{r.get('gt_calls',0)} |")
lines += ['', '## Telemetry Snapshot', '']
for k,v in sorted(events.items(), key=lambda kv:(-kv[1],kv[0]))[:30]:
    lines.append(f'- {k}: {v}')
lines += ['', '## Decision', '', '- Do not update `state/best_track2_api_version.json`.', '- Do not run `validation_B_holdout` or final from this candidate.', '- Next useful work: order-specific process candidate synthesis and retail process-aware candidate preservation. Current broad-scan trimming helps micro but not enough for top1 readiness.', '']
report.write_text('\n'.join(lines), encoding='utf-8')
# candidate state only, not protected best
candidate={
 'version': VERSION,
 'run_id': RUN,
 'stage': 'validation_A_medium',
 'split': 'validation_A_limit30_materialized_from_V9_4_5',
 'joint_success': summary.get('joint',0),
 'result_success': summary.get('result',0),
 'tool_success': summary.get('tool',0),
 'micro_tool_accuracy': summary.get('micro',0),
 'valid': summary.get('valid',0),
 'protected_best_updated': False,
 'final_submission': False,
 'note': 'Candidate only. Does not beat protected V6; no holdout/final.',
 'updated_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
}
(CODEX/'state'/'candidate_track2_api_version.json').write_text(json.dumps(candidate, ensure_ascii=False, indent=2), encoding='utf-8')
with (CODEX/'README_STATUS.md').open('a', encoding='utf-8') as f:
    f.write(f"\n## V9.5 Frozen A Medium {now}\n\n")
    f.write(f"- Report: `{report}`\n")
    f.write(f"- run_id: `{RUN}`\n")
    f.write(f"- valid: {summary.get('valid',0)}, joint: {summary.get('joint',0):.4f}, result: {summary.get('result',0):.4f}, tool: {summary.get('tool',0):.4f}, micro: {summary.get('micro',0):.4f}\n")
    f.write("- Split hygiene fixed with materialized validation_A_limit30; protected best unchanged; no final submission.\n")
print(report)
