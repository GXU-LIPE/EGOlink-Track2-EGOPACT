#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build V10 full dev/offline memory bank without final hidden metadata."""
from __future__ import annotations

import hashlib, json, os, re, statistics, time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

CODEX = Path(os.environ.get('CODEX_ROOT','/home/data-gxu/acm/egolink2026-main/code/track2/codex'))
EGO = Path(os.environ.get('EGO_ROOT','/home/data-gxu/acm/egolink2026-main/code/track2/EgoBench'))
OUT = CODEX / 'memory_bank_v10'
FINAL_IDS = {'retail6','retail10','kitchen4','restaurant5','order2'}
SCENARIOS = ('retail','kitchen','restaurant','order')


def read_json(path: Path):
    try: return json.loads(path.read_text(encoding='utf-8'))
    except Exception: return None


def write_json(path: Path, data: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def write_jsonl(path: Path, rows: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def scenario_id(path: Path) -> str:
    return path.stem.lower()


def scenario_from_id(sid: str) -> str:
    for s in SCENARIOS:
        if sid.startswith(s): return s
    return 'global'


def infer_type(text: str) -> str:
    low=(text or '').lower()
    checks=[('replace',['replace','swap','instead','change']),('remove',['remove','delete','cancel','take out']),('add',['add','put','buy','order','include','cart']),('payment/tax',['total','payment','pay','tax','amount','cost','price']),('nutrition',['nutrition','calorie','protein','fat','sodium','carb','sugar']),('recipe/menu/fridge',['recipe','ingredient','fridge','freezer','stock','shopping list','menu']),('compare/filter',['lowest','highest','cheapest','healthiest','compare','least','most','filter']),('visual/pointing',['point','shown','visible','image','video','left','right','label','menu'])]
    for name,words in checks:
        if any(w in low for w in words): return name
    return 'general'


def tool_family(name: str) -> str:
    n=(name or '').lower()
    if n.startswith(('compute_total','tally_total')): return 'aggregate_compute'
    if n.startswith(('add','create')) or '_to_' in n: return 'state_changing_add'
    if n.startswith(('remove','delete','clear')) or '_from_' in n: return 'state_changing_remove'
    if n.startswith(('update','modify','set_')): return 'state_changing_update'
    if n.startswith(('get_','find_','list_','search_','retrieve_','check_')) or 'summary' in n or 'current' in n: return 'read_only_retrieval'
    return 'other'


def load_tool_schemas():
    tools={}
    for scen in SCENARIOS:
        p=EGO/'tools'/scen/f'{scen}_tools.json'
        data=read_json(p) or []
        for t in data:
            name=t.get('name') or t.get('function',{}).get('name')
            if not name: continue
            params=((t.get('parameters') or {}).get('properties') if isinstance(t.get('parameters'),dict) else None) or ((t.get('function',{}).get('parameters') or {}).get('properties') if isinstance(t.get('function'),dict) else {}) or {}
            req=(t.get('parameters') or {}).get('required') if isinstance(t.get('parameters'),dict) else None
            if req is None and isinstance(t.get('function'),dict): req=(t.get('function',{}).get('parameters') or {}).get('required')
            tools[name]={'scenario':scen,'family':tool_family(name),'required_params':req or [],'param_names':list(params.keys())}
    return tools


def build_tool_constitution(tools):
    out={}
    for name,t in tools.items():
        family=t['family']; scen=t['scenario']
        risky=[]; pre=[]; post=[]; rewrite=[]
        if family.startswith('state_changing'):
            pre += ['retrieve uncertain entity first','pin user_id when required','pin restaurant_name for order/restaurant when applicable','canonicalize entity name']
            post += ['record successful mutation ledger','do not repeat same mutation']
            risky += ['duplicate mutation','wrong user_id','wrong restaurant/menu','dish/set_meal confusion']
        if family=='aggregate_compute':
            pre += ['use confirmed current state or retrieved items','run near the end when requested']
            risky += ['aggregate too early','same-parameter aggregate loop','invented item list']
        if scen=='retail': rewrite += ['narrow candidates before numeric attribute sweeps']
        if scen=='order': rewrite += ['order aggregate dishes[] use product_name + quantity','separate dish_name and set_meal_name']
        out[name]={**t,'entity_type':guess_entity(name),'preconditions':pre,'postconditions':post,'risky_misuse':risky,'safe_rewrite':rewrite}
    return out


def guess_entity(name):
    n=name.lower()
    for e in ['product','dish','set_meal','recipe','ingredient','category','restaurant','order','cart','shopping_list','menu']:
        if e in n: return e
    return 'general'


def collect_tasks():
    rows=[]; excluded=[]
    for root in [EGO/'scenarios'/'final', EGO/'scenarios']:
        if not root.exists(): continue
        for p in sorted(root.glob('*.json')):
            sid=scenario_id(p)
            data=read_json(p)
            if not isinstance(data,list): continue
            if sid in FINAL_IDS:
                excluded.append({'file':str(p),'reason':'official_final_scenario_excluded','tasks':len(data)})
                continue
            for i,item in enumerate(data,1):
                instr=str(item.get('Instruction') or item.get('instruction') or '')
                gt=item.get('ground_truth') or item.get('GroundTruth') or item.get('tool_calls') or item.get('reference')
                rows.append({'source_file':str(p),'scenario_id':sid,'scenario':scenario_from_id(sid),'idx':i,'task_id':item.get('task_id',i),'task_type':infer_type(instr),'has_gt':gt is not None,'instruction':instr,'ground_truth':gt})
    # de-dup by scenario_id::idx
    dedup={}
    for r in rows:
        dedup.setdefault(f"{r['scenario_id']}::{r['idx']}", r)
    return list(dedup.values()), excluded


def gt_tool_names(gt):
    calls=[]
    if isinstance(gt,dict):
        for key in ['tool_calls','tools','actions','trajectory']:
            if isinstance(gt.get(key),list): calls=gt[key]; break
    elif isinstance(gt,list): calls=gt
    names=[]
    for c in calls:
        if isinstance(c,dict): names.append(str(c.get('tool_name') or c.get('name') or c.get('tool') or ''))
    return [n for n in names if n]


def build_process_templates(tasks):
    templates={}
    base={
      'retail':['identify visible/filter candidate','narrow by category/origin/brand/taste','retrieve numeric attributes only for narrowed candidates','mutate cart/list if requested','compute aggregate only if requested','final response'],
      'order':['pin restaurant/user','inspect current order/menu','resolve dish vs set meal','add/remove/replace once','compute tax/payment/nutrition if requested','final response'],
      'kitchen':['identify recipe/current state','get recipe ingredients once','determine branch from menu/fridge/stock/list','apply menu/shopping-list mutation','compute total nutritions from confirmed ingredients','final response'],
      'restaurant':['identify dish/set meal/menu item','retrieve requested menu attributes','mutate order if requested','compute nutrition/payment if requested','final response']}
    for scen,steps in base.items():
        templates[scen]={'scenario':scen,'soft_template':steps,'do_not_use_as_hard_fsm':True}
    return templates


def build_cards(tasks, tools):
    scoring=[{'card_id':'v10::scoring::joint_success','card_type':'scoring_rule','scenario':'global','task_type':'general','text':'Track2 is not ordinary QA. Joint success requires both final DB result correctness and required tool process coverage. Micro is diagnostic. Output exactly one format: JSON tool array or natural language, never both. In final_eval never use hidden scenario JSON metadata.','no_final_metadata':True}]
    failures=[]
    patterns=[('visual_followup_when_grounding_expected','Do not ask the user for visible product/dish/category names when retrieval/tools can narrow candidates.'),('broad_retail_scan','Avoid catalog-wide price/tax/discount/nutrition sweeps; narrow candidates first.'),('order_process_mismatch','Order result-only success can fail tool success; inspect order/menu, mutate once, then compute requested aggregate.'),('dish_set_meal_confusion','Keep dish_name and set_meal_name separate; use set meal tools for set meals.'),('missing_retrieval_before_mutation','Retrieve uncertain entity/user/restaurant before state-changing calls.'),('missing_final_aggregate','If total/tax/payment/nutrition is requested, compute near the end.'),('aggregate_loop','Do not repeat same-parameter aggregate calls.'),('duplicate_mutation','Never repeat successful same-object state-changing mutation.'),('kitchen_branch_quantity_failure','Kitchen branch decisions need recipe ingredients and confirmed quantities/locations.'),('mixed_json_text','Tool turns must be pure JSON array; message turns must be natural language only.')]
    for i,(pid,text) in enumerate(patterns,1): failures.append({'card_id':f'v10::failure::{pid}','card_type':'failure_pattern','scenario':'global','task_type':'general','text':text,'no_final_metadata':True})
    success=[]
    seq_counter=Counter(); scen_counter=Counter(); type_counter=Counter()
    covered=0
    for t in tasks:
        names=gt_tool_names(t.get('ground_truth'))
        if names:
            fams=[tool_family(n) for n in names]
            seq=' -> '.join([f for f in fams if f])
            seq_counter[(t['scenario'],t['task_type'],seq)]+=1; covered+=1
            scen_counter[t['scenario']]+=1; type_counter[t['task_type']]+=1
    for idx,((scen,tt,seq),cnt) in enumerate(seq_counter.most_common(300),1):
        success.append({'card_id':f'v10::success::{idx:04d}','card_type':'success_trajectory','scenario':scen,'task_type':tt,'text':f'Abstract successful/GT process pattern seen {cnt} times: {seq}. Use as process-shape prior only, not as an answer shortcut.','count':cnt,'no_final_metadata':True})
    canonical=[]
    canonical_rules=[('global','Use canonical DB names after retrieval; strip whitespace and fix case conservatively.'),('order','Order aggregate dishes[] uses product_name + quantity; mutation tools use dish_name or set_meal_name as required.'),('order','For Annie Italian Restaurant, category alias Steaks -> Selected Steaks and Pasta -> Italian Pasta when exact category is empty.'),('restaurant','set_meal_name must remain set_meal_name; do not send set meals to dish remove tools.'),('retail','Retail product_name must be canonical; ranking words are filters, not automatic aggregate triggers.')]
    for i,(scen,text) in enumerate(canonical_rules,1): canonical.append({'card_id':f'v10::canonical::{i:03d}','card_type':'canonicalization','scenario':scen,'task_type':'general','text':text,'no_final_metadata':True})
    return scoring, failures, success, canonical, {'covered_by_gt':covered,'scenario_gt_cards':dict(scen_counter),'task_type_gt_cards':dict(type_counter)}


def build_index(cards):
    docs=[]; df=Counter(); total=0
    for c in cards:
        text=' '.join(str(c.get(k,'')) for k in ['card_id','card_type','scenario','task_type','text'])
        toks=re.findall(r'[a-zA-Z0-9_]+', text.lower())
        tf=Counter(toks); total+=len(toks)
        for tok in tf: df[tok]+=1
        docs.append({'card_id':c['card_id'],'length':len(toks),'tf':dict(tf)})
    return {'num_docs':len(docs),'avgdl':(total/len(docs) if docs else 1),'df':dict(df),'docs':docs}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tools=load_tool_schemas(); tasks, excluded=collect_tasks()
    tool_const=build_tool_constitution(tools)
    templates=build_process_templates(tasks)
    scoring, failures, success, canonical, stats=build_cards(tasks, tools)
    # Add process/tool constitution cards to unified retrieval cards.
    process_cards=[{'card_id':f'v10::process::{scen}','card_type':'process_template','scenario':scen,'task_type':'general','text':' -> '.join(data['soft_template']),'no_final_metadata':True} for scen,data in templates.items()]
    tool_cards=[]
    for scen in SCENARIOS:
        fams=Counter(v['family'] for v in tool_const.values() if v['scenario']==scen)
        tool_cards.append({'card_id':f'v10::tool_constitution::{scen}','card_type':'tool_constitution','scenario':scen,'task_type':'general','text':f'{scen} tools by family: '+', '.join(f'{k}={v}' for k,v in sorted(fams.items()))+'. Respect required params and preconditions.','no_final_metadata':True})
    visual_cards=[{'card_id':f'v10::visual::{scen}','card_type':'visual_grounding','scenario':scen,'task_type':'visual/pointing','text':'Use retrieval-first grounding when visual evidence is ambiguous; do not ask visual follow-up in final_eval.','no_final_metadata':True} for scen in ['retail','order','restaurant','kitchen']]
    all_cards=scoring+process_cards+tool_cards+failures+success+canonical+visual_cards
    write_json(OUT/'tool_constitution.json', tool_const)
    write_json(OUT/'process_templates.json', templates)
    write_jsonl(OUT/'scoring_rule_cards.jsonl', scoring)
    write_jsonl(OUT/'failure_pattern_cards.jsonl', failures)
    write_jsonl(OUT/'success_trajectory_cards.jsonl', success)
    write_jsonl(OUT/'canonicalization_cards.jsonl', canonical)
    write_jsonl(OUT/'visual_grounding_cards.jsonl', visual_cards)
    idx=defaultdict(lambda: defaultdict(int))
    for t in tasks: idx[t['scenario']][t['task_type']]+=1
    write_json(OUT/'scenario_task_type_index.json', {s:dict(v) for s,v in idx.items()})
    audit={'generated_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'dev_offline_task_total':len(tasks),'tasks_with_gt_or_tool_trajectory':sum(1 for t in tasks if t['has_gt']),'tasks_entered_memory':stats['covered_by_gt'],'uncovered_task_count':len(tasks)-stats['covered_by_gt'],'uncovered_reason':'No explicit ground_truth/tool trajectory found; still counted in task type index but not success trajectory cards.','excluded_final_files':excluded,'card_counts':{'scoring':len(scoring),'failure':len(failures),'success':len(success),'canonicalization':len(canonical),'process':len(process_cards),'tool_constitution':len(tool_cards),'visual':len(visual_cards),'total_retrieval_cards':len(all_cards)},'scenario_card_counts':Counter(c.get('scenario','global') for c in all_cards),'task_type_card_counts':Counter(c.get('task_type','general') for c in all_cards),'used_final_hidden_metadata':False,'final_scenarios_excluded':sorted(FINAL_IDS)}
    audit['scenario_card_counts']=dict(audit['scenario_card_counts']); audit['task_type_card_counts']=dict(audit['task_type_card_counts'])
    write_json(OUT/'memory_coverage_audit.json', audit)
    manifest={'memory_bank':'memory_bank_v10','generated_at':audit['generated_at'],'source_policy':'dev/offline/history only; official final scenario files excluded from memory construction','files':{},'used_final_hidden_metadata':False}
    for p in sorted(OUT.glob('*')):
        if p.is_file(): manifest['files'][p.name]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
    emb=OUT/'embeddings'; emb.mkdir(exist_ok=True)
    write_jsonl(emb/'cards.jsonl', all_cards)
    write_json(emb/'simple_bm25_index.json', build_index(all_cards))
    manifest['files']['embeddings/cards.jsonl']={'sha256':hashlib.sha256((emb/'cards.jsonl').read_bytes()).hexdigest(),'bytes':(emb/'cards.jsonl').stat().st_size}
    manifest['files']['embeddings/simple_bm25_index.json']={'sha256':hashlib.sha256((emb/'simple_bm25_index.json').read_bytes()).hexdigest(),'bytes':(emb/'simple_bm25_index.json').stat().st_size}
    write_json(OUT/'memory_bank_manifest.json', manifest)
    ts=time.strftime('%Y%m%d_%H%M%S')
    report=CODEX/'reports'/f'V10_FULL_MEMORY_BANK_BUILD_{ts}.md'
    lines=[f'# V10 Full Memory Bank Build {ts}','',f'- output: `{OUT}`',f'- dev/offline task total: {len(tasks)}',f'- tasks with GT/tool trajectory: {audit["tasks_with_gt_or_tool_trajectory"]}',f'- tasks entered success memory: {stats["covered_by_gt"]}',f'- uncovered task count: {audit["uncovered_task_count"]}',f'- used final hidden metadata: no',f'- final scenarios excluded: {", ".join(sorted(FINAL_IDS))}','', '## Card Counts','']
    for k,v in audit['card_counts'].items(): lines.append(f'- {k}: {v}')
    lines += ['', '## Scenario Card Counts','']
    for k,v in sorted(audit['scenario_card_counts'].items()): lines.append(f'- {k}: {v}')
    report.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    audit_report=CODEX/'reports'/f'V10_MEMORY_COVERAGE_AUDIT_{ts}.md'
    audit_report.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'report':str(report),'audit_report':str(audit_report),'audit':audit}, ensure_ascii=False, indent=2))

if __name__=='__main__': main()
