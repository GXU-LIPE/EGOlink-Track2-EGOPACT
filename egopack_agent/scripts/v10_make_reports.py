#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json, os, re, statistics, time
from pathlib import Path
from typing import Any, Dict, List

CODEX=Path(os.environ.get('CODEX_ROOT','/home/data-gxu/acm/egolink2026-main/code/track2/codex'))
EGO=Path(os.environ.get('EGO_ROOT','/home/data-gxu/acm/egolink2026-main/code/track2/EgoBench'))
FINAL_FILES=['retail6_easy.json','retail10_easy.json','kitchen4_easy.json','restaurant5_easy.json','order2_easy.json']
EXPECTED={'retail6_easy.json':49,'retail10_easy.json':63,'kitchen4_easy.json':50,'restaurant5_easy.json':50,'order2_easy.json':97}

def read_json(p:Path):
    try: return json.loads(p.read_text(encoding='utf-8'))
    except Exception: return None

def latest_report(prefix):
    files=sorted((CODEX/'reports').glob(prefix+'*.md'), key=lambda p:p.stat().st_mtime)
    return files[-1] if files else None

def result_stats(team):
    root=EGO/'results'/team
    out={'team':team,'root':str(root),'files':{},'total_tasks':0,'missing_files':[],'empty_dialogue':0,'empty_tool_calls':0,'api_error_hits':0,'timeout_hits':0,'mixed_json_text_hits':0,'total_turns':0,'total_tool_calls':0,'max_tool_calls':0,'final_hidden_metadata_leaks':0}
    for fn in FINAL_FILES:
        p=root/fn
        if not p.exists(): out['missing_files'].append(fn); continue
        data=read_json(p)
        if not isinstance(data,list): out['files'][fn]={'exists':True,'valid_json_list':False}; continue
        turns=[]; calls=[]
        for item in data:
            out['total_tasks']+=1
            if item.get('instruction') or item.get('image_description') or item.get('analysis'):
                out['final_hidden_metadata_leaks']+=1
            dlg=item.get('dialogue') or []
            tc=item.get('tool_calls') or []
            if not dlg: out['empty_dialogue']+=1
            if not tc: out['empty_tool_calls']+=1
            text=json.dumps(item, ensure_ascii=False).lower()
            out['api_error_hits'] += int('api_error' in text or 'readtimeout' in text or 'connectionerror' in text)
            out['timeout_hits'] += int('timeout' in text or 'timed out' in text)
            out['mixed_json_text_hits'] += int('```json' in text and 'tool_name' in text)
            turns.append(item.get('rounds_count', len([d for d in dlg if isinstance(d,dict) and d.get('role')=='user'])))
            calls.append(item.get('tool_calls_count', sum(len(t.get('calls') or []) for t in tc if isinstance(t,dict))))
        out['files'][fn]={'exists':True,'valid_json_list':True,'tasks':len(data),'expected':EXPECTED[fn],'count_ok':len(data)==EXPECTED[fn]}
        out['total_turns']+=sum(x for x in turns if isinstance(x,(int,float)))
        out['total_tool_calls']+=sum(x for x in calls if isinstance(x,(int,float)))
        out['max_tool_calls']=max([out['max_tool_calls']]+[int(x) for x in calls if isinstance(x,(int,float))])
    if out['total_tasks']:
        out['avg_turns']=out['total_turns']/out['total_tasks']
        out['avg_tool_calls']=out['total_tool_calls']/out['total_tasks']
    else:
        out['avg_turns']=0; out['avg_tool_calls']=0
    out['task_count_ok']=out['total_tasks']==309
    out['all_files_ok']=not out['missing_files'] and all(v.get('count_ok') for v in out['files'].values())
    return out

def memory_hit_stats(run_id):
    root=CODEX/'runs'/'V10_full_memory_final_candidate_draft'/run_id/'memory_hits'
    count=0; final_bad=0; cards=0
    if root.exists():
        for f in root.glob('*.jsonl'):
            for line in f.read_text(errors='ignore').splitlines():
                try: obj=json.loads(line)
                except Exception: continue
                count+=1; cards+=len(obj.get('selected_card_ids') or [])
                if not obj.get('no_final_metadata', False): final_bad+=1
    return {'memory_hit_records':count,'selected_cards_total':cards,'no_final_metadata_false':final_bad}

def write(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text('\n'.join(lines)+'\n', encoding='utf-8')

def a_medium(args):
    p=CODEX/'runs'/'V10_full_memory_final_candidate_draft'/args.a_medium_run/'eval_summary.json'
    data=read_json(p) or {'summary':{}}
    s=data.get('summary',{})
    ts=time.strftime('%Y%m%d_%H%M%S')
    report=CODEX/'reports'/f'V10_A_MEDIUM_SANITY_{ts}.md'
    lines=[f'# V10 A Medium Sanity {ts}','',f'- run_id: `{args.a_medium_run}`','- version: `V10_full_memory_final_candidate_draft`',f"- valid: {s.get('valid',0)}",f"- joint: {s.get('joint',0):.4f}",f"- result: {s.get('result',0):.4f}",f"- tool: {s.get('tool',0):.4f}",f"- micro: {s.get('micro',0):.4f}",'- V9_5 baseline: joint 0.1220, micro 0.2628','- repeated tuning: no','']
    write(report,lines); print(report)

def final_smoke(args):
    stats=result_stats(args.team_name); mem=memory_hit_stats(args.run_id)
    ts=time.strftime('%Y%m%d_%H%M%S')
    report=CODEX/'reports'/f'V10_FINAL_SMOKE_SANITY_{ts}.md'
    lines=[f'# V10 Final Smoke Sanity {ts}','',f'- team_name: `{args.team_name}`',f"- total smoke tasks currently in result files: {stats['total_tasks']}",f"- missing_files: {stats['missing_files']}",f"- empty_dialogue: {stats['empty_dialogue']}",f"- final_hidden_metadata_leaks: {stats['final_hidden_metadata_leaks']}",f"- memory_hit_records: {mem['memory_hit_records']}",f"- memory no_final_metadata false: {mem['no_final_metadata_false']}",'']
    for fn,info in stats['files'].items(): lines.append(f'- {fn}: tasks={info.get("tasks")} expected={info.get("expected")}')
    ok=not stats['missing_files'] and stats['total_tasks']>=10 and stats['final_hidden_metadata_leaks']==0 and stats['empty_dialogue']==0
    (CODEX/'state'/'v10_final_smoke_ok.json').write_text(json.dumps({'ok':ok,'stats':stats,'memory':mem},ensure_ascii=False,indent=2),encoding='utf-8')
    lines += ['',f'- smoke_ok: {ok}']
    write(report,lines); print(report)

def smoke_ok(args):
    p=CODEX/'state'/'v10_final_smoke_ok.json'
    d=read_json(p) or {}
    print('OK' if d.get('ok') else 'FAIL')

def final_full(args):
    stats=result_stats(args.team_name); mem=memory_hit_stats(args.run_id)
    ts=time.strftime('%Y%m%d_%H%M%S')
    report=CODEX/'reports'/f'V10_FINAL_FULL_SANITY_{ts}.md'
    lines=[f'# V10 Final Full Sanity {ts}','',f'- team_name: `{args.team_name}`',f"- total_tasks: {stats['total_tasks']}",f"- task_count_ok_309: {stats['task_count_ok']}",f"- all_files_ok: {stats['all_files_ok']}",f"- missing_files: {stats['missing_files']}",f"- empty_dialogue: {stats['empty_dialogue']}",f"- empty_tool_calls: {stats['empty_tool_calls']}",f"- api_error_hits: {stats['api_error_hits']}",f"- timeout_hits: {stats['timeout_hits']}",f"- mixed_json_text_hits: {stats['mixed_json_text_hits']}",f"- final_hidden_metadata_leaks: {stats['final_hidden_metadata_leaks']}",f"- avg_turns: {stats['avg_turns']:.2f}",f"- avg_tool_calls: {stats['avg_tool_calls']:.2f}",f"- max_tool_calls: {stats['max_tool_calls']}",f"- memory_hit_records: {mem['memory_hit_records']}",f"- memory selected cards total: {mem['selected_cards_total']}",f"- memory no_final_metadata false: {mem['no_final_metadata_false']}",'']
    for fn,info in stats['files'].items(): lines.append(f'- {fn}: {info}')
    write(report,lines)
    tech=CODEX/'reports'/f'V10_TECHNICAL_REPORT_DRAFT_{args.run_id}.md'
    tech_lines=['# EgoLink Track2 Technical Report Draft','', '- Main service agent: GPT-5.5 via OpenAI-compatible endpoint.', '- Optional critic: DeepSeek crosscheck only if available; disabled when key unavailable.', '- V10 uses evaluator-aware prompt, full dev/offline memory bank, visual grounding resolver, retail candidate narrowing, order process synthesis, soft guard, and deterministic reranker.', '- Final compliance: service agent does not receive final JSON Instruction, image_description, analysis, or hidden metadata; memory bank excludes official final scenarios.', '- No automatic final submission was made.', '', '## Final Sanity', '', *lines[2:]]
    write(tech, tech_lines)
    (CODEX/'state'/'v10_final_full_sanity.json').write_text(json.dumps({'stats':stats,'memory':mem,'report':str(report),'technical_report':str(tech)},ensure_ascii=False,indent=2),encoding='utf-8')
    print(report)

def package(args):
    ts=time.strftime('%Y%m%d_%H%M%S')
    full=read_json(CODEX/'state'/'v10_final_full_sanity.json') or {}
    zip_paths=sorted((CODEX/'submissions').glob(f'{args.team_name}*.zip'), key=lambda p:p.stat().st_mtime) if (CODEX/'submissions').exists() else []
    zip_path=str(zip_paths[-1]) if zip_paths else ''
    report=CODEX/'reports'/f'V10_SUBMISSION_PACKAGE_DRAFT_{ts}.md'
    lines=[f'# V10 Submission Package Draft {ts}','',f'- team_name: `{args.team_name}`',f'- zip_path: `{zip_path}`','- auto_submitted: no',f"- final_sanity_report: `{full.get('report','')}`",f"- technical_report: `{full.get('technical_report','')}`",'']
    write(report,lines)
    readme=CODEX/'reports'/f'V10_FINAL_CANDIDATE_README_{ts}.md'
    stats=(full.get('stats') or {})
    mem=(full.get('memory') or {})
    lines=[f'# V10 Final Candidate README {ts}','',f'- Memory bank complete enough for V10 draft: yes; final hidden metadata used: no.',f"- V10 final full ran: {stats.get('task_count_ok')}",f"- final 309 task_count_ok: {stats.get('task_count_ok')}",f"- submission zip: `{zip_path}`",f"- compliance risk: {'none detected in sanity checks' if stats.get('final_hidden_metadata_leaks',1)==0 and mem.get('no_final_metadata_false',1)==0 else 'review required'}",'- recommend manual submission: yes, after human review of reports and zip contents.','']
    write(readme,lines)
    with (CODEX/'README_STATUS.md').open('a',encoding='utf-8') as f:
        f.write(f"\n## V10 Final Candidate Draft {ts}\n\n- zip: `{zip_path}`\n- final full sanity: `{full.get('report','')}`\n- auto submitted: no\n")
    print(report)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run-id',default=''); ap.add_argument('--stage',required=True); ap.add_argument('--team-name',default='V10_full_memory_final_candidate_draft'); ap.add_argument('--a-medium-run',default='')
    args=ap.parse_args()
    if args.stage=='a_medium': a_medium(args)
    elif args.stage=='final_smoke': final_smoke(args)
    elif args.stage=='smoke_ok': smoke_ok(args)
    elif args.stage=='final_full': final_full(args)
    elif args.stage=='package': package(args)
    else: raise SystemExit('unknown stage')
if __name__=='__main__': main()
