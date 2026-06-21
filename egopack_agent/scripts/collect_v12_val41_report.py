#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, re, statistics, time
from pathlib import Path

CODEX=Path('/home/data-gxu/acm/egolink2026-main/code/track2/codex')
EGO=Path('/home/data-gxu/acm/egolink2026-main/code/track2/EgoBench')


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding='utf-8', errors='replace'))
    except Exception:
        return None


def count_logs(run_dir: Path, pattern: str) -> int:
    n=0
    for p in (run_dir/'logs').glob('*.log'):
        try:
            n += len(re.findall(pattern, p.read_text(encoding='utf-8', errors='replace'), flags=re.I))
        except Exception:
            pass
    return n


def load_jsonl(path: Path):
    try:
        for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
            line=line.strip()
            if not line: continue
            try:
                yield json.loads(line)
            except Exception:
                continue
    except Exception:
        return


def qwen3vl_hit_stats(run_dir: Path) -> dict:
    hit_dir=run_dir/'qwen3vl_grounding_hits'
    stats={
        'hit_files': 0,
        'events': 0,
        'task_ids': set(),
        'with_top_k_events': 0,
        'grounding_failed_events': 0,
        'missing_events': 0,
        'video_fallback_events': 0,
        'status_counts': {},
    }
    if not hit_dir.exists():
        stats['task_ids']=[]
        return stats
    stats['hit_files']=len(list(hit_dir.glob('*.jsonl')))
    for p in sorted(hit_dir.glob('*.jsonl')):
        for row in load_jsonl(p) or []:
            stats['events']+=1
            tid=str(row.get('task_id') or p.stem)
            stats['task_ids'].add(tid)
            status=str(row.get('cache_status') or '')
            stats['status_counts'][status]=stats['status_counts'].get(status,0)+1
            if int(row.get('top_k_count') or 0)>0:
                stats['with_top_k_events']+=1
            if row.get('grounding_failed'):
                stats['grounding_failed_events']+=1
            if status=='missing':
                stats['missing_events']+=1
            if row.get('video_fallback_used'):
                stats['video_fallback_events']+=1
    stats['task_ids']=sorted(stats['task_ids'], key=lambda x: int(x) if x.isdigit() else x)
    return stats


def task_rows(data: dict) -> list[dict]:
    if not isinstance(data, dict): return []
    for key in ['tasks','task_results','results','all_results','per_task_results']:
        v=data.get(key)
        if isinstance(v, list): return [x for x in v if isinstance(x, dict)]
    rows=[]
    for key,val in data.items():
        if isinstance(val, dict) and any(k in val for k in ['joint','result','tool','micro','scenario']):
            row=dict(val); row.setdefault('id',key); rows.append(row)
    return rows


def scenario_of(row: dict) -> str:
    for k in ['scenario','scenario_name']:
        if row.get(k): return str(row[k])
    tid=str(row.get('task_id') or row.get('uid') or row.get('id') or '')
    m=re.search(r'(retail|kitchen|restaurant|order)', tid)
    return m.group(1) if m else 'unknown'


def boolish(v):
    if isinstance(v,bool): return v
    if isinstance(v,(int,float)): return v >= 1
    if isinstance(v,str): return v.lower() in {'true','1','yes','success'}
    return False


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--version', required=True)
    ap.add_argument('--model', default='gpt-5.5')
    args=ap.parse_args()
    run_dir=CODEX/'runs'/args.version/args.run_id
    data=load_json(run_dir/'eval_summary.json') or {}
    summary=data.get('summary') or {}
    rows=task_rows(data)
    by_s={}
    for row in rows:
        s=scenario_of(row)
        d=by_s.setdefault(s, {'n':0,'joint':0,'result':0,'tool':0,'micro_vals':[]})
        d['n']+=1
        d['joint']+=int(boolish(row.get('joint') if 'joint' in row else row.get('joint_success')))
        d['result']+=int(boolish(row.get('result') if 'result' in row else row.get('result_success')))
        d['tool']+=int(boolish(row.get('tool') if 'tool' in row else row.get('tool_success')))
        mv=row.get('micro') or row.get('micro_accuracy') or row.get('task_accuracy')
        if isinstance(mv,(int,float)): d['micro_vals'].append(float(mv))
    visual_followups=count_logs(run_dir, r'visual_followup|need the exact name|can.t see|what.*dish name|what.*product name')
    broad_scan=count_logs(run_dir, r'broad_scan|full.*scan|scan.*all|all.*nutrition')
    api_errors=count_logs(run_dir, r'api error|direct api error|traceback|failed after|429|timeout')
    qwen_prompt_hits=qwen3vl_hit_stats(run_dir)
    qwen_hits=[]
    for p in sorted((CODEX/'visual_cache_v12/qwen3vl_grounding').glob('*.json')):
        d=load_json(p) or {}
        if d.get('status'):
            qwen_hits.append({'file':p.name,'status':d.get('status'),'teacher':d.get('teacher'),'video':d.get('video_path','')})
    report=CODEX/'reports'/f'V12_VAL41_QWEN3VL_MEMORY_{args.run_id}.md'
    md=[]
    md.append('# V12 Val41 Qwen3VL Memory Report')
    md.append('')
    md.append(f'- generated_at: `{time.strftime("%Y-%m-%dT%H:%M:%S%z")}`')
    md.append(f'- version: `{args.version}`')
    md.append(f'- run_id: `{args.run_id}`')
    md.append(f'- model: `{args.model}`')
    md.append('- stage: `validation_A_medium` frozen/materialized val41')
    md.append('- final_run: `false`')
    md.append('- v10_zip_overwritten: `false`')
    md.append('')
    md.append('## Summary')
    for k in ['valid','joint','result','tool','micro','avg_task_accuracy','correct_calls','gt_calls','interaction_calls']:
        if k in summary:
            md.append(f'- {k}: `{summary[k]}`')
    if not summary:
        md.append('- status: `summary_missing`')
    md.append(f'- api_errors: `{api_errors}`')
    md.append(f'- visual_followup_mentions: `{visual_followups}`')
    md.append(f'- broad_scan_mentions: `{broad_scan}`')
    md.append(f'- qwen3vl_prompt_hit_files: `{qwen_prompt_hits.get("hit_files",0)}`')
    md.append(f'- qwen3vl_prompt_events: `{qwen_prompt_hits.get("events",0)}`')
    md.append(f'- qwen3vl_prompt_tasks: `{len(qwen_prompt_hits.get("task_ids",[]))}`')
    md.append(f'- qwen3vl_events_with_top_k: `{qwen_prompt_hits.get("with_top_k_events",0)}`')
    md.append(f'- qwen3vl_video_fallback_events: `{qwen_prompt_hits.get("video_fallback_events",0)}`')
    md.append(f'- qwen3vl_missing_events: `{qwen_prompt_hits.get("missing_events",0)}`')
    md.append('')
    md.append('## Scenario Breakdown')
    md.append('| scenario | n | joint | result | tool | avg_micro |')
    md.append('|---|---:|---:|---:|---:|---:|')
    for s,d in sorted(by_s.items()):
        avg=sum(d['micro_vals'])/len(d['micro_vals']) if d['micro_vals'] else 0.0
        md.append(f"| {s} | {d['n']} | {d['joint']} | {d['result']} | {d['tool']} | {avg:.4f} |")
    if not by_s:
        md.append('| none | 0 | 0 | 0 | 0 | 0 |')
    md.append('')
    md.append('## Qwen3VL Grounding Cache')
    md.append(f'- cached_cards: `{len(qwen_hits)}`')
    for item in qwen_hits[:20]:
        md.append(f"- `{item['file']}` status=`{item['status']}` teacher=`{item['teacher']}`")
    md.append('')
    md.append('## Qwen3VL Prompt Hit Audit')
    md.append(f"- status_counts: `{json.dumps(qwen_prompt_hits.get('status_counts',{}), ensure_ascii=False)}`")
    md.append(f"- task_ids_with_hits: `{', '.join(qwen_prompt_hits.get('task_ids',[])[:80])}`")
    md.append('')
    md.append('## Decision')
    joint=float(summary.get('joint') or 0.0)
    micro=float(summary.get('micro') or 0.0)
    if joint >= 0.122 or micro >= 0.2949:
        md.append('- V12 val41 is at least comparable to V10 A_medium on one of the tracked criteria; inspect scenario breakdown before any final expansion.')
    else:
        md.append('- V12 val41 is below V10 A_medium on both joint and micro; do not expand.')
    if joint >= 0.15 and micro >= 0.30:
        md.append('- Meets the requested improvement target for a stronger candidate; next step can be validation_B_holdout or targeted order repair.')
    else:
        md.append('- Does not cleanly meet the stronger target; order/visual process remains the primary risk.')
    report.write_text('\n'.join(md)+'\n', encoding='utf-8')
    print(report)

if __name__=='__main__':
    main()
