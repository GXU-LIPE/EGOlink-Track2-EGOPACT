#!/usr/bin/env python3
import argparse, difflib, hashlib, json, os, shutil, subprocess, time
from pathlib import Path
from urllib.request import Request, urlopen
BASE='https://raw.githubusercontent.com/ego-link/egolink2026/main/code/track2/EgoBench'
FILES=['README.md','run_all_scenarios.sh','run/multi_agent.py','analysis_scripts/evaluate_interaction.py','analysis_scripts/run_eval.sh','scenarios/final/retail6.json','scenarios/final/retail10.json','scenarios/final/kitchen4.json','scenarios/final/restaurant5.json','scenarios/final/order2.json']
FINAL=['retail6.json','retail10.json','kitchen4.json','restaurant5.json','order2.json']
def sha(p):
    h=hashlib.sha256();
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()
def txt(p):
    try: return Path(p).read_text(encoding='utf-8',errors='replace')
    except Exception: return ''
def dl(rel,out):
    for k in list(os.environ):
        if k.lower() in {'http_proxy','https_proxy','all_proxy'}: os.environ.pop(k,None)
    if out.exists() and out.stat().st_size>0: return 'cached'
    req=Request(f'{BASE}/{rel}',headers={'User-Agent':'codex-track2-final-sync'})
    last=''
    for i in range(1,5):
        try:
            with urlopen(req,timeout=60) as r: data=r.read()
            if not data: raise RuntimeError('empty')
            out.parent.mkdir(parents=True,exist_ok=True); out.write_bytes(data)
            return f'downloaded {len(data)} bytes'
        except Exception as e:
            last=f'{type(e).__name__}: {e}'; time.sleep(i*2)
    raise RuntimeError(f'download failed {rel}: {last}')
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--ego',required=True); ap.add_argument('--codex',required=True); ap.add_argument('--apply',action='store_true'); args=ap.parse_args()
    ego=Path(args.ego).resolve(); codex=Path(args.codex).resolve(); ts=time.strftime('%Y%m%d_%H%M%S')
    snap=codex/'official_snapshots'/f'final_min_{ts}'/'code'/'track2'/'EgoBench'
    for d in ['reports','analysis','patches','backups','logs']: (codex/d).mkdir(parents=True,exist_ok=True)
    log=[]
    for rel in FILES:
        msg=dl(rel,snap/rel); log.append(f'{rel}: {msg}')
    rows=[]; diffs=[]; missing=[]
    for rel in FILES:
        lp=ego/rel; rp=snap/rel
        if not lp.exists(): status='missing_local'; missing.append(rel)
        else: status='same' if sha(lp)==sha(rp) else 'different'
        if status=='different': diffs.append(rel)
        rows.append((rel,status,lp.stat().st_size if lp.exists() else '',rp.stat().st_size,sha(lp) if lp.exists() else '',sha(rp)))
    broot=codex/'backups'/f'official_final_sync_{ts}'
    applied=[]
    if args.apply:
        for rel,status,*_ in rows:
            if status=='same': continue
            lp=ego/rel; rp=snap/rel
            if lp.exists():
                bp=broot/rel; bp.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(lp,bp)
            lp.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(rp,lp); applied.append(rel)
        sh=ego/'run_all_scenarios.sh'
        if sh.exists(): sh.chmod(sh.stat().st_mode|0o755)
    csv=codex/'analysis'/f'official_final_min_compare_{ts}.csv'
    csv.write_text('file,status,local_size,official_size,local_sha,official_sha\n'+'\n'.join(','.join('"'+str(x).replace('"','""')+'"' for x in r) for r in rows)+'\n',encoding='utf-8')
    diffp=codex/'patches'/f'official_final_min_diff_{ts}.diff'
    with diffp.open('w',encoding='utf-8') as f:
        for rel in diffs:
            lp=ego/rel if not args.apply else broot/rel; rp=snap/rel
            if lp.exists() and lp.stat().st_size<2_000_000 and rp.stat().st_size<2_000_000:
                f.writelines(difflib.unified_diff(txt(lp).splitlines(True),txt(rp).splitlines(True),fromfile='before/'+rel,tofile='official/'+rel)); f.write('\n')
            else:
                f.write(f'large or missing diff skipped {rel}\n')
    checks=[]
    for rel in ['run/multi_agent.py','analysis_scripts/evaluate_interaction.py']:
        cp=subprocess.run(['python3','-m','py_compile',str(ego/rel)],capture_output=True,text=True); checks.append((rel,cp.returncode,cp.stderr[-300:]))
    cp=subprocess.run(['bash','-n',str(ego/'run_all_scenarios.sh')],capture_output=True,text=True); checks.append(('run_all_scenarios.sh',cp.returncode,cp.stderr[-300:]))
    readme=txt(ego/'README.md')
    final_present={f:(ego/'scenarios'/'final'/f).exists() for f in FINAL}
    best={}
    bp=codex/'state'/'best_track2_api_version.json'
    if bp.exists():
        try: best=json.loads(bp.read_text(encoding='utf-8'))
        except Exception as e: best={'error':str(e)}
    report=codex/'reports'/f'OFFICIAL_FINAL_MIN_SYNC_{ts}.md'
    report.write_text('\n'.join([
        f'# Official Final Minimal Sync {ts}','',f'- Source: `{BASE}`',f'- Applied: `{args.apply}`',f'- Snapshot: `{snap}`',f'- Compare CSV: `{csv}`',f'- Diff: `{diffp}`','',
        '## File Status','',*[f'- `{r[0]}`: {r[1]}' for r in rows],'','## Applied','',*([f'- `{x}`' for x in applied] or ['- none']),'','## Final README Markers','',
        f'- contains `--final_eval`: `{"--final_eval" in readme}`',f'- contains `309`: `{"309" in readme}`',f'- contains final scenario IDs: `{all(x in readme for x in ["retail6_easy","retail10_easy","kitchen4_easy","restaurant5_easy","order2_easy"])}`',f'- contains final output names: `{all(x in readme for x in ["retail_easy.json","kitchen_easy.json","restaurant_easy.json","order_easy.json"])}`',f'- contains direct final JSON cheating warning: `{("directly accessed" in readme.lower() or "cheating" in readme.lower())}`','',
        '## Final Files Present','',*[f'- `{k}`: `{v}`' for k,v in final_present.items()],'','## Validation','',*[f'- `{a}` rc={b} {c}' for a,b,c in checks],'','## Current Best Context','',*[f'- {k}: `{v}`' for k,v in best.items() if k in {'version','run_id','joint_success','result_success','tool_success','micro_tool_accuracy','avg_tool_calls','note'}],'','## No Auto Submit','- No final submission was made.'])+'\n',encoding='utf-8')
    final_guide=codex/'reports'/f'FINAL_STAGE_SUBMISSION_GUIDE_{ts}.md'
    final_guide.write_text('\n'.join(['# Final Stage Submission Guide '+ts,'','- Final phase: official README now states final tasks were released 2026-06-16 20:00 GMT+8 and due 2026-06-22 20:00 GMT+8.','- Final set: 309 tasks across `retail6_easy`, `retail10_easy`, `kitchen4_easy`, `restaurant5_easy`, `order2_easy`.','- Run with `--final_eval`.','- Required results live in `results/{team_name}/`: `retail_easy.json`, `kitchen_easy.json`, `restaurant_easy.json`, `order_easy.json`, and `{team_name}.pdf`.','- Service agent must not directly read `scenarios/final/*.json`; use only official interaction pipeline/video/simulated user/tool observations.','- Do not auto-submit final.'])+'\n',encoding='utf-8')
    top=codex/'reports'/f'TOP1_READINESS_ANALYSIS_{ts}.md'
    top.write_text('\n'.join(['# Top1 Readiness Analysis '+ts,'','## Verdict','- Not top1-ready yet. Current best gate is 50% joint on 4 selected dev tasks; final has 309 held-out tasks and ranking is joint-success driven.','- Result-only success will not win because tool trajectory/process coverage is part of joint success.','', '## Current Best', *[f'- {k}: `{v}`' for k,v in best.items() if k in {'version','run_id','joint_success','result_success','tool_success','micro_tool_accuracy','avg_tool_calls','model','endpoint','note'}], '', '## Main Risks','- order: DB result can pass while process/tool trajectory mismatches.','- kitchen: recipe branch and nutrition/tool loops remain weak.','- final compliance: wrappers must avoid hidden final JSON reads.','', '## Recommended Path','- Keep `V6_1_3_gpt55_guarded_endpoint` as base.','- Run final-mode smoke on one task per final scenario after this sync.','- Patch order/kitchen helpers only if smoke confirms the same failure classes.','- Expand to 10-20 per final scenario, then package but do not submit.'])+'\n',encoding='utf-8')
    with (codex/'README_STATUS.md').open('a',encoding='utf-8') as f: f.write(f'\n## Official Final Minimal Sync {ts}\n\n- Report: `{report}`\n- Final guide: `{final_guide}`\n- Top1 analysis: `{top}`\n- Applied: {", ".join(applied) if applied else "none"}\n- No final submission was made.\n')
    (codex/'logs'/f'official_final_min_sync_{ts}.log').write_text('\n'.join(log)+'\n',encoding='utf-8')
    print(json.dumps({'ts':ts,'report':str(report),'final_guide':str(final_guide),'top1':str(top),'applied':applied,'diffs':diffs,'missing':missing,'checks':checks,'final_present':final_present},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
