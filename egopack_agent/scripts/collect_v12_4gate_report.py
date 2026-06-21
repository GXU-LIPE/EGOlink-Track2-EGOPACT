#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, re, statistics, time
from pathlib import Path

CODEX=Path('/home/data-gxu/acm/egolink2026-main/code/track2/codex')
EGO=Path('/home/data-gxu/acm/egolink2026-main/code/track2/EgoBench')
TASKS=[('retail9','retail9_easy_eval.json'),('restaurant4','restaurant4_easy_eval.json'),('order1','order1_easy_eval.json'),('kitchen2','kitchen2_easy_eval.json')]
BASELINES=[
 ('V6_1_3','gpt-5.5-V6_1_3_gpt55_guarded_endpoint-gpt55_endpoint_gate_20260617_105936'),
 ('V6_1_5','gpt-5.5-V6_1_5_kitchen_branch_repair-gpt55_next_gate_20260617_125219'),
 ('V10_A_medium','not_4gate'),
]

def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding='utf-8', errors='replace'))
    except Exception:
        return None

def scalar_metrics(data):
    if not isinstance(data, dict): return {}
    keys=['joint_success_rate','result_based_success_rate','tool_based_success_rate','micro_tool_call_accuracy','avg_tool_calls']
    out={k:data.get(k) for k in keys if k in data}
    for alt in ['joint','result','tool','micro']:
        if alt in data: out[alt]=data[alt]
    return out

def bool_metric(data, names):
    if not isinstance(data, dict): return None
    for n in names:
        if n in data: return data[n]
    return None

def per_eval(model_name: str):
    ed=EGO/'eval_result'/model_name
    rows=[]
    totals={'joint':0,'result':0,'tool':0,'matched':0,'gt':0,'count':0}
    for spec,fn in TASKS:
        data=load_json(ed/fn)
        row={'spec':spec,'path':str(ed/fn),'exists':(ed/fn).exists()}
        if isinstance(data, dict):
            row.update(scalar_metrics(data))
            # tolerate several evaluator schema variants
            j=bool_metric(data,['joint_success','joint_success_rate'])
            r=bool_metric(data,['result_success','result_based_success','result_based_success_rate'])
            t=bool_metric(data,['tool_success','tool_based_success','tool_based_success_rate'])
            row['raw_keys']=sorted(list(data.keys()))[:40]
            for k,v in data.items():
                if isinstance(v,(int,float,bool,str)) and k not in row and len(str(v))<80:
                    row[k]=v
            # Try list/detail records.
            details=data.get('details') or data.get('per_task_results') or data.get('tasks') or []
            if isinstance(details,list) and details:
                d=details[0]
                if isinstance(d,dict):
                    for src,dst in [('joint_success','joint_success'),('result_success','result_success'),('tool_success','tool_success'),('matched_tool_calls','matched_tool_calls'),('gt_tool_calls','gt_tool_calls')]:
                        if src in d: row[dst]=d[src]
            totals['count']+=1
            if isinstance(row.get('joint_success'), bool): totals['joint']+=int(row['joint_success'])
            if isinstance(row.get('result_success'), bool): totals['result']+=int(row['result_success'])
            if isinstance(row.get('tool_success'), bool): totals['tool']+=int(row['tool_success'])
            for mk in ['matched_tool_calls','matched_tools']:
                if isinstance(row.get(mk), int): totals['matched']+=row[mk]; break
            for gk in ['gt_tool_calls','total_gt_tool_calls']:
                if isinstance(row.get(gk), int): totals['gt']+=row[gk]; break
        rows.append(row)
    return rows, totals

def load_result_tool_calls(model_name: str, spec: str):
    p=EGO/'results'/model_name/f'{spec}_easy.json'
    data=load_json(p)
    calls=[]; rounds=None; tool_count=None; empty=False
    if isinstance(data,list) and data:
        item=data[0]
        rounds=item.get('rounds_count')
        tool_count=item.get('tool_calls_count')
        for block in item.get('tool_calls',[]) or []:
            for call in block.get('calls',[]) or []:
                if isinstance(call,dict):
                    calls.append({'tool_name':call.get('tool_name'), 'parameters':call.get('parameters')})
        empty=(not item.get('dialogue'))
    return {'path':str(p),'rounds':rounds,'tool_calls_count':tool_count,'calls':calls,'empty_dialogue':empty}

def load_grounding(spec: str):
    p=CODEX/'visual_cache_v12/qwen3vl_grounding'/f'{spec}_1.json'
    d=load_json(p) or {}
    hit=False
    text=json.dumps(d,ensure_ascii=False).lower()
    if spec.startswith('retail'):
        hit=any(x in text for x in ['cheese','italy','red wrapper','trapezoid'])
    elif spec.startswith('restaurant'):
        hit=any(x in text for x in ['salmon','steak','set'])
    elif spec.startswith('order'):
        hit=any(x in text for x in ['annie','restaurant','menu','dish'])
    elif spec.startswith('kitchen'):
        hit=any(x in text for x in ['pork','fried','cutlet','chop'])
    return {'exists':p.exists(),'path':str(p),'status':d.get('status'),'teacher':d.get('teacher'),'error':d.get('error','')[:180],'key_entity_hit':hit,'top_k':d.get('top_k_candidates',[])[:5], 'uncertainty':d.get('uncertainty_notes',[])[:3]}

def memory_hits(version: str, run_id: str, spec: str):
    # task_id is always 1 for this gate; file naming uses task id only.
    p=CODEX/'runs'/version/run_id/'memory_hits'/'1.jsonl'
    rows=[]
    if p.exists():
        for line in p.read_text(encoding='utf-8',errors='replace').splitlines():
            try:
                obj=json.loads(line)
                if obj.get('scenario') and spec.startswith(obj.get('scenario')):
                    rows.append(obj)
            except Exception:
                pass
    return {'path':str(p),'count':len(rows),'last_card_ids':(rows[-1].get('selected_card_ids') if rows else [])}

def count_in_file(path: Path, pattern: str):
    if not path.exists(): return 0
    return len(re.findall(pattern, path.read_text(encoding='utf-8',errors='replace'), flags=re.I))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--version', required=True)
    ap.add_argument('--model', default='gpt-5.5')
    args=ap.parse_args()
    model_name=f'{args.model}-{args.version}-{args.run_id}'
    rows, totals=per_eval(model_name)
    summary_data = load_json(EGO/'eval_result'/model_name/'summary.json') or {}
    official_summary = summary_data.get('summary') if isinstance(summary_data, dict) else {}
    official_rows = {}
    if isinstance(summary_data, dict):
        for item in summary_data.get('all_results', []) or []:
            if isinstance(item, dict):
                key = f"{item.get('scenario')}{item.get('scenario_number')}"
                official_rows[key] = item
    task_details=[]
    tool_counts=[]
    api_errors=0
    visual_followups=0
    broad_scan=0
    for spec,_ in TASKS:
        res=load_result_tool_calls(model_name,spec)
        if isinstance(res.get('tool_calls_count'),int): tool_counts.append(res['tool_calls_count'])
        log=CODEX/'runs'/args.version/args.run_id/'logs'/f'{spec}.log'
        api_errors+=count_in_file(log, r'api error|direct api error|traceback|failed after')
        visual_followups+=count_in_file(log, r'visual_followup|what.*visible|which.*image|provide.*label')
        broad_scan+=count_in_file(log, r'broad_scan|retail_broad_scan|full.*scan')
        task_details.append({
            'spec':spec,
            'result':res,
            'grounding':load_grounding(spec),
            'memory':memory_hits(args.version,args.run_id,spec),
        })
    report=CODEX/'reports'/f'V12_4GATE_QWEN3VL_MEMORY_{args.run_id}.md'
    md=[]
    md.append('# V12 4-Gate Qwen3VL Memory Report')
    md.append('')
    md.append(f'- generated_at: `{time.strftime("%Y-%m-%dT%H:%M:%S%z")}`')
    md.append(f'- version: `{args.version}`')
    md.append(f'- run_id: `{args.run_id}`')
    md.append(f'- model_name: `{model_name}`')
    md.append('- scope: 4-gate only; final was not run; V10 submission zip was not touched')
    md.append('')
    md.append('## Aggregate Metrics')
    if official_summary:
        joint=float(official_summary.get('avg_joint_success_rate') or 0.0)
        result=float(official_summary.get('avg_result_based_success_rate') or 0.0)
        tool=float(official_summary.get('avg_tool_based_success_rate') or 0.0)
        micro=official_summary.get('micro_accuracy')
        md.append(f'- joint: `{joint:.4f}`')
        md.append(f'- result: `{result:.4f}`')
        md.append(f'- tool: `{tool:.4f}`')
        md.append(f'- micro: `{float(micro):.4f}`' if micro is not None else '- micro: `unknown`')
    else:
        c=max(totals['count'],1)
        joint=totals['joint']/c
        result=totals['result']/c
        tool=totals['tool']/c
        micro=(totals['matched']/totals['gt']) if totals['gt'] else None
        md.append(f'- joint: `{joint:.4f}` ({totals["joint"]}/{c})')
        md.append(f'- result: `{result:.4f}` ({totals["result"]}/{c})')
        md.append(f'- tool: `{tool:.4f}` ({totals["tool"]}/{c})')
        md.append(f'- micro: `{micro:.4f}` ({totals["matched"]}/{totals["gt"]})' if micro is not None else '- micro: `unknown`')
    md.append(f'- avg_tool_calls: `{statistics.mean(tool_counts):.2f}`' if tool_counts else '- avg_tool_calls: `unknown`')
    md.append(f'- max_tool_calls: `{max(tool_counts)}`' if tool_counts else '- max_tool_calls: `unknown`')
    md.append(f'- api_errors: `{api_errors}`')
    md.append(f'- visual_followup_mentions: `{visual_followups}`')
    md.append(f'- broad_scan_mentions: `{broad_scan}`')
    md.append('')
    md.append('## Per-Task Metrics')
    md.append('| task | joint | result | tool | micro/raw | tool_calls | qwen3vl_status | key_entity_hit | memory_hits |')
    md.append('|---|---:|---:|---:|---|---:|---|---:|---:|')
    row_by={r['spec']:r for r in rows}
    detail_by={d['spec']:d for d in task_details}
    for spec,_ in TASKS:
        r=row_by.get(spec,{})
        official = official_rows.get(spec, {})
        d=detail_by.get(spec,{})
        res=d.get('result',{})
        g=d.get('grounding',{})
        mem=d.get('memory',{})
        micro_raw=''
        for k in ['micro_tool_call_accuracy','tool_call_accuracy','matched_tool_calls']:
            if k in r: micro_raw=str(r[k]); break
        joint_cell = official.get('joint_success_rate', r.get('joint_success', r.get('joint_success_rate','')))
        result_cell = official.get('result_based_success_rate', r.get('result_success', r.get('result_based_success_rate','')))
        tool_cell = official.get('tool_based_success_rate', r.get('tool_success', r.get('tool_based_success_rate','')))
        micro_cell = official.get('micro_accuracy', micro_raw)
        md.append(f"| {spec} | `{joint_cell}` | `{result_cell}` | `{tool_cell}` | `{micro_cell}` | `{res.get('tool_calls_count','')}` | `{g.get('status','')}` | `{g.get('key_entity_hit')}` | `{mem.get('count',0)}` |")
    md.append('')
    md.append('## Tool Trajectories')
    for d in task_details:
        md.append(f"### {d['spec']}")
        calls=d['result'].get('calls') or []
        if not calls:
            md.append('- no tool calls recorded')
        for i,call in enumerate(calls,1):
            params=json.dumps(call.get('parameters',{}),ensure_ascii=False)
            md.append(f"{i}. `{call.get('tool_name')}` `{params[:500]}`")
        md.append(f"- qwen3vl: status=`{d['grounding'].get('status')}`, key_entity_hit=`{d['grounding'].get('key_entity_hit')}`, error=`{d['grounding'].get('error','')}`")
        md.append(f"- memory_last_card_ids: `{d['memory'].get('last_card_ids',[])}`")
        md.append('')
    md.append('## Baseline Comparison')
    md.append('- V6_1_3 protected best 4-gate reference from prior report: joint 50%, result 75%, tool 50%, micro 70.83%.')
    md.append('- V10 validated on frozen A_medium, not directly 4-gate in this report: joint 12.20%, micro 29.49%.')
    if joint >= 0.5 and (micro is None or float(micro) >= 0.7083):
        rec='V12 is competitive with V6_1_3 on 4-gate; next step can be frozen val41.'
    elif joint >= 0.5 or (micro is not None and float(micro) >= 0.7083):
        rec='V12 has partial 4-gate signal; inspect per-task regressions before frozen val41.'
    else:
        rec='Do not expand yet; V12 underperforms the protected 4-gate baseline or lacks tool-process evidence.'
    md.append(f'- recommendation: {rec}')
    md.append('')
    md.append('## Compliance Notes')
    md.append('- final_eval was not run.')
    md.append('- V10 zip was not overwritten.')
    md.append('- Qwen3-VL cards are cached visual teacher evidence only and do not call EgoBench tools.')
    report.write_text('\n'.join(md),encoding='utf-8')
    print(report)

if __name__=='__main__':
    main()
