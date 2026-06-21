#!/usr/bin/env python3
import argparse, difflib, hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

BASE = 'https://raw.githubusercontent.com/ego-link/egolink2026/main/code/track2/EgoBench'
FILES = [
    'README.md',
    'run_all_scenarios.sh',
    'run/multi_agent.py',
    'run/prompts.py',
    'run/utils.py',
    'analysis_scripts/evaluate_interaction.py',
    'analysis_scripts/run_eval.sh',
    'analysis_scripts/analyze_error_reasons.py',
    'config/service_agent_config.py',
    'config/user_agent_config.py',
    'tools/retail/retail_db.py',
    'tools/retail/retail_init.py',
    'tools/retail/retail_tools.json',
    'tools/order/order_db.py',
    'tools/order/order_init.py',
    'tools/order/order_tools.json',
    'tools/restaurant/restaurant_db.py',
    'tools/restaurant/restaurant_init.py',
    'tools/restaurant/restaurant_tools.json',
    'tools/kitchen/kitchen_db.py',
    'tools/kitchen/kitchen_init.py',
    'tools/kitchen/kitchen_tools.json',
    'scenarios/final/retail6.json',
    'scenarios/final/retail10.json',
    'scenarios/final/kitchen4.json',
    'scenarios/final/restaurant5.json',
    'scenarios/final/order2.json',
]
FINAL_FILES = ['retail6.json','retail10.json','kitchen4.json','restaurant5.json','order2.json']
APPLY_FILES = [
    'README.md',
    'run_all_scenarios.sh',
    'run/multi_agent.py',
    'analysis_scripts/evaluate_interaction.py',
    'analysis_scripts/run_eval.sh',
    'analysis_scripts/analyze_error_reasons.py',
    'scenarios/final/retail6.json',
    'scenarios/final/retail10.json',
    'scenarios/final/kitchen4.json',
    'scenarios/final/restaurant5.json',
    'scenarios/final/order2.json',
]

def read(p):
    try:
        return Path(p).read_text(encoding='utf-8', errors='replace')
    except Exception:
        return ''

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()

def fetch(rel, out, log):
    out=Path(out)
    if out.exists() and out.stat().st_size > 0:
        return True, 'cached'
    for k in list(os.environ):
        if k.lower() in {'http_proxy','https_proxy','all_proxy'}:
            os.environ.pop(k, None)
    url=f'{BASE}/{rel}'
    req=Request(url, headers={'User-Agent':'codex-track2-official-audit'})
    last=''
    for attempt in range(1,4):
        try:
            with urlopen(req, timeout=45) as r:
                data=r.read()
            if not data:
                raise RuntimeError('empty response')
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(data)
            return True, f'downloaded {len(data)} bytes'
        except Exception as e:
            last=f'{type(e).__name__}: {e}'
            log.append(f'WARN download {rel} attempt {attempt}: {last}')
            time.sleep(2*attempt)
    return False, last

def markers(text):
    return {
        'final_eval_flag': '--final_eval' in text,
        'final_309_tasks': '309' in text,
        'final_task_set': all(x in text for x in ['retail6_easy','retail10_easy','kitchen4_easy','restaurant5_easy','order2_easy']),
        'submission_jsons': all(x in text for x in ['retail_easy.json','kitchen_easy.json','restaurant_easy.json','order_easy.json']),
        'technical_report_pdf': ('.pdf' in text.lower() and 'technical' in text.lower()),
        'no_direct_final_json': ('directly accessed' in text.lower() or 'cheating' in text.lower() or 'scenarios/final' in text),
    }

def copy_with_backup(src, dst, backup_root):
    src=Path(src); dst=Path(dst)
    backup=None
    if dst.exists():
        backup=backup_root / dst.relative_to(backup_root.parents[1] / 'EgoBench') if False else backup_root / dst.relative_to(EGO)
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dst, backup)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(backup) if backup else None

# patched after argparse: module-global for copy_with_backup
EGO = None

def main():
    global EGO
    ap=argparse.ArgumentParser()
    ap.add_argument('--ego', required=True)
    ap.add_argument('--codex', required=True)
    ap.add_argument('--apply', action='store_true')
    args=ap.parse_args()
    EGO=Path(args.ego).resolve(); codex=Path(args.codex).resolve()
    ts=time.strftime('%Y%m%d_%H%M%S')
    snap=codex/'official_snapshots'/f'raw_min_{ts}'/'code'/'track2'/'EgoBench'
    logs=[]
    for rel in FILES:
        ok,msg=fetch(rel, snap/rel, logs)
        logs.append(f'{"OK" if ok else "FAIL"} {rel}: {msg}')
    rows=[]; diffs=[]; missing=[]
    for rel in FILES:
        lp=EGO/rel; rp=snap/rel
        if not rp.exists():
            rows.append((rel,'download_failed','','','',''))
            continue
        if not lp.exists():
            status='missing_local'; missing.append(rel)
        else:
            status='same' if sha(lp)==sha(rp) else 'different'
            if status=='different': diffs.append(rel)
        rows.append((rel,status,str(lp.stat().st_size) if lp.exists() else '',str(rp.stat().st_size),sha(lp) if lp.exists() else '',sha(rp)))
    analysis=codex/'analysis'; reports=codex/'reports'; patches=codex/'patches'; backups=codex/'backups'; logsdir=codex/'logs'
    for d in [analysis,reports,patches,backups,logsdir]: d.mkdir(parents=True, exist_ok=True)
    csvp=analysis/f'official_update_file_compare_{ts}.csv'
    with csvp.open('w',encoding='utf-8') as f:
        f.write('file,status,local_size,official_size,local_sha,official_sha\n')
        for r in rows:
            f.write(','.join('"'+str(x).replace('"','""')+'"' for x in r)+'\n')
    diffp=patches/f'official_update_diff_{ts}.diff'
    with diffp.open('w',encoding='utf-8') as f:
        for rel in diffs:
            lp=EGO/rel; rp=snap/rel
            if lp.stat().st_size > 2_000_000 or rp.stat().st_size > 2_000_000:
                f.write(f'--- remote/{rel}\n+++ official/{rel}\n@@ large/binary skipped: local {lp.stat().st_size}, official {rp.stat().st_size}\n\n')
                continue
            f.writelines(difflib.unified_diff(read(lp).splitlines(True), read(rp).splitlines(True), fromfile=f'remote/{rel}', tofile=f'official/{rel}'))
            f.write('\n')
    applied=[]; backups_made=[]
    if args.apply:
        broot=backups/f'official_update_{ts}'
        for rel in APPLY_FILES:
            rp=snap/rel; lp=EGO/rel
            if not rp.exists():
                continue
            if lp.exists() and sha(lp)==sha(rp):
                continue
            backup=None
            if lp.exists():
                backup=broot/rel
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(lp, backup)
                backups_made.append(str(backup))
            lp.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(rp, lp)
            applied.append(rel)
        # Make shell script executable if applied/downloaded
        sh=EGO/'run_all_scenarios.sh'
        if sh.exists():
            sh.chmod(sh.stat().st_mode | 0o755)
    # syntax checks after apply
    checks=[]
    for rel in ['run/multi_agent.py','analysis_scripts/evaluate_interaction.py','analysis_scripts/analyze_error_reasons.py']:
        p=EGO/rel
        if p.exists():
            cp=subprocess.run(['python3','-m','py_compile',str(p)], text=True, capture_output=True)
            checks.append((rel,cp.returncode,cp.stderr.strip()[-500:]))
    if (EGO/'run_all_scenarios.sh').exists():
        cp=subprocess.run(['bash','-n',str(EGO/'run_all_scenarios.sh')], text=True, capture_output=True)
        checks.append(('run_all_scenarios.sh',cp.returncode,cp.stderr.strip()[-500:]))
    remote_readme=read(EGO/'README.md'); official_readme=read(snap/'README.md')
    local_markers=markers(remote_readme); official_markers=markers(official_readme)
    final_present={f:(EGO/'scenarios'/'final'/f).exists() for f in FINAL_FILES}
    bestp=codex/'state'/'best_track2_api_version.json'
    try: best=json.loads(bestp.read_text(encoding='utf-8')) if bestp.exists() else {}
    except Exception as e: best={'error':str(e)}
    report=reports/f'OFFICIAL_UPDATE_AUDIT_{ts}.md'
    final_report=reports/f'FINAL_STAGE_SUBMISSION_GUIDE_{ts}.md'
    top1_report=reports/f'TOP1_READINESS_ANALYSIS_{ts}.md'
    report.write_text('\n'.join([
        f'# Official Update Audit {ts}', '',
        f'- Remote EgoBench: `{EGO}`',
        f'- Raw official snapshot: `{snap}`',
        f'- Download source: `{BASE}`',
        f'- Compare CSV: `{csvp}`',
        f'- Diff preview: `{diffp}`',
        f'- Applied updates: `{args.apply}`', '',
        '## Summary', '',
        f'- Different files: {len(diffs)}',
        f'- Missing local files: {len(missing)}',
        f'- Applied files: {len(applied)}',
        '', '## Differences', '',
        *[f'- different: `{x}`' for x in diffs],
        *[f'- missing local: `{x}`' for x in missing],
        '', '## Applied', '',
        *([f'- `{x}`' for x in applied] or ['- none']),
        '', '## Final Markers', '',
        *[f'- official {k}: `{v}`' for k,v in official_markers.items()],
        *[f'- remote_after {k}: `{v}`' for k,v in local_markers.items()],
        '', '## Required Final Scenario Files Present', '',
        *[f'- `{k}`: `{v}`' for k,v in final_present.items()],
        '', '## Validation', '',
        *[f'- `{rel}`: rc={rc} {err}' for rel,rc,err in checks],
        '', '## Notes', '',
        '- Official code sync was limited to documentation, run/eval scripts, and final scenario JSONs. Codex wrappers and previous run outputs were not modified.',
        '- Service agent still must not directly inspect final scenario JSON content; these files are only for official runner/simulated user/evaluator flow.',
        '- No final submission was made.',
        '', '## Download Log Tail', '',
        *[f'- {x}' for x in logs[-80:]],
    ])+'\n', encoding='utf-8')
    final_report.write_text('\n'.join([
        f'# Final Stage Submission Guide {ts}', '',
        '- Official latest README says final tasks were released on 2026-06-16 20:00 GMT+8 and submission deadline is 2026-06-22 20:00 GMT+8.',
        '- Final evaluation contains 309 tasks across `retail6_easy`, `retail10_easy`, `kitchen4_easy`, `restaurant5_easy`, and `order2_easy`.',
        '- Run final inference with `--final_eval` so the runner uses the final scenario set.',
        '- Required archive layout is under `results/{team_name}/` and includes `retail_easy.json`, `kitchen_easy.json`, `restaurant_easy.json`, `order_easy.json`, plus `{team_name}.pdf`.',
        '- The technical report PDF should describe methodology, model usage, prompts/guards, API usage, and reproducibility.',
        '- It is explicitly not allowed for the service agent to directly access `scenarios/final/*.json`; final JSON may only be consumed by the official runner/simulated user/evaluator pipeline.',
        '- Do not auto-submit final from automation; only generate package/report under codex.',
        '', '## Immediate Impact On Our Pipeline', '',
        '- `run_all_scenarios.sh` and `run/multi_agent.py` must support final mode and the five held-out scenario files.',
        '- Our codex runner must continue using copied runner plus wrapper, but final mode must avoid dev-GT and direct final JSON shortcuts in prompt/visual prior code.',
        '- Previous reports that expected `kitchen.json/order.json/restaurant.json/retail.json` should be superseded by the README-observed `*_easy.json` layout.',
    ])+'\n', encoding='utf-8')
    top=[]
    top += [f'# Top1 Readiness Analysis {ts}', '', '## Current Best', '']
    for k in ['version','run_id','joint_success','result_success','tool_success','micro_tool_accuracy','avg_tool_calls','model','endpoint','external_api_used','note']:
        if k in best: top.append(f'- {k}: `{best[k]}`')
    top += ['', '## Verdict', '', '- Not top1-ready yet.', '- The best validated gate is 50% joint on four handpicked dev tasks. A top1 final solution over 309 held-out tasks likely needs much higher and more stable joint success, especially for order and kitchen.', '- Track2 ranking emphasizes `avg_joint_success_rate`, so result-only success is insufficient; missing process tools or mismatched trajectory shape still hurts.', '', '## Why Current Plan Can Still Improve', '', '- GPT-5.5 endpoint is live and schema/guard wrappers are already useful.', '- Retail duplicate-mutation and restaurant paths are relatively stable on the gate.', '- Official final scenarios now narrow the target to five scenario IDs, so targeted final-mode validation is possible once we run without direct final JSON leakage.', '', '## Top1 Blockers', '', '- Order: trajectory/process alignment still trails DB result correctness.', '- Kitchen: recipe branch control and nutrition computation still cause long or mismatched tool traces.', '- Final compliance: visual priors and wrappers must not read final scenario JSON directly as hidden labels.', '- Evidence: no broad 10-20 per-scenario final-like eval has shown high joint success yet.', '', '## Recommended Next Move', '', '- Keep `V6_1_3_gpt55_guarded_endpoint` as the base candidate.', '- Sync official final files/scripts, then run a final-mode smoke with `--final_eval --num_tasks 1` for each of the five final scenario IDs.', '- Patch only order/kitchen trajectory helpers based on dev and allowed simulated-user/video signals.', '- Expand to 10-20 tasks per final scenario before creating the final package.', '- Generate package under `codex/submissions`, but do not submit automatically.']
    top1_report.write_text('\n'.join(top)+'\n', encoding='utf-8')
    status=codex/'README_STATUS.md'
    with status.open('a',encoding='utf-8') as f:
        f.write('\n'.join(['',f'## Official Final Update {ts}','',f'- Audit: `{report}`',f'- Final guide: `{final_report}`',f'- Top1 analysis: `{top1_report}`',f'- Applied official files: {", ".join(applied) if applied else "none"}',f'- Current top1 verdict: not ready; best 4-task joint={best.get("joint_success","unknown")}.','- No final submission was made.'])+'\n')
    (logsdir/f'official_min_sync_{ts}.log').write_text('\n'.join(logs)+'\n',encoding='utf-8')
    print(json.dumps({'ts':ts,'report':str(report),'final_report':str(final_report),'top1_report':str(top1_report),'diffs':diffs,'missing':missing,'applied':applied,'checks':checks,'final_present':final_present}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
