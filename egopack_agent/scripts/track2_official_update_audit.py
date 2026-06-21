#!/usr/bin/env python3
import argparse, csv, difflib, filecmp, hashlib, json, os, re, shutil, subprocess, sys
from pathlib import Path
from datetime import datetime

TRACKED = [
    'README.md',
    'run_all_scenarios.sh',
    'run/multi_agent.py',
    'analysis_scripts/evaluate_interaction.py',
    'tools/kitchen.py', 'tools/order.py', 'tools/restaurant.py', 'tools/retail.py',
    'tools/__init__.py',
]
DIRS = ['run', 'analysis_scripts', 'tools', 'scenarios/final']

def sha256(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()

def safe_read(p):
    try:
        return Path(p).read_text(encoding='utf-8', errors='replace')
    except Exception:
        return ''

def list_files(root):
    out=[]
    root=Path(root)
    for d in DIRS:
        dd=root/d
        if dd.exists():
            for p in dd.rglob('*'):
                if p.is_file() and not any(part in {'.git','__pycache__'} for part in p.parts):
                    out.append(str(p.relative_to(root)))
    for f in TRACKED:
        if (root/f).is_file() and f not in out:
            out.append(f)
    return sorted(set(out))

def contains(path, pats):
    text=safe_read(path)
    return {k: bool(re.search(v, text, re.I|re.S)) for k,v in pats.items()}

def get_best(codex):
    p=Path(codex)/'state'/'best_track2_api_version.json'
    if p.exists():
        try: return json.loads(p.read_text())
        except Exception as e: return {'error':str(e)}
    return {}

def newest_reports(codex):
    reports=Path(codex)/'reports'
    names=['GPT55_ENDPOINT_GATE_SUMMARY','GPT55_NEXT_GATE_SUMMARY','HUMAN_PRIOR_GATE_SUMMARY','ORDER_PROCESS_ALIGNER','KITCHEN_BRANCH_REPAIR','VISUAL_PRIOR_ABLATION']
    found=[]
    if reports.exists():
        for n in names:
            ms=sorted(reports.glob(n+'*.md'), key=lambda p:p.stat().st_mtime, reverse=True)
            if ms: found.append(str(ms[0]))
    return found

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--ego', required=True)
    ap.add_argument('--snapshot', required=True)
    ap.add_argument('--codex', required=True)
    ap.add_argument('--timestamp', required=True)
    ap.add_argument('--report', required=True)
    ap.add_argument('--final-report', required=True)
    ap.add_argument('--top1-report', required=True)
    args=ap.parse_args()
    ego=Path(args.ego); snap=Path(args.snapshot); codex=Path(args.codex)
    snap_repo=snap.parents[2] if len(snap.parents) >= 3 else snap
    try:
        commit=subprocess.check_output(['git','-C',str(snap_repo),'rev-parse','HEAD'], text=True).strip()
    except Exception:
        commit='unknown'
    files=sorted(set(list_files(ego)) | set(list_files(snap)))
    rows=[]
    changed=[]; missing_local=[]; extra_local=[]
    for rel in files:
        lp=ego/rel; rp=snap/rel
        if lp.exists() and rp.exists():
            lsha=sha256(lp); rsha=sha256(rp); same=lsha==rsha
            rows.append({'file':rel,'status':'same' if same else 'different','local_sha':lsha,'official_sha':rsha,'local_size':lp.stat().st_size,'official_size':rp.stat().st_size})
            if not same: changed.append(rel)
        elif rp.exists():
            rows.append({'file':rel,'status':'missing_local','local_sha':'','official_sha':sha256(rp),'local_size':'','official_size':rp.stat().st_size})
            missing_local.append(rel)
        elif lp.exists():
            rows.append({'file':rel,'status':'extra_local','local_sha':sha256(lp),'official_sha':'','local_size':lp.stat().st_size,'official_size':''})
            extra_local.append(rel)
    csvp=codex/'analysis'/f'official_update_file_compare_{args.timestamp}.csv'
    with csvp.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=['file','status','local_sha','official_sha','local_size','official_size'])
        w.writeheader(); w.writerows(rows)
    # create diff for changed core files, but do not auto-apply yet
    diffp=codex/'patches'/f'official_update_diff_{args.timestamp}.diff'
    with diffp.open('w', encoding='utf-8') as out:
        for rel in changed:
            lp=ego/rel; rp=snap/rel
            # skip large json/video/data files
            if lp.stat().st_size > 2_000_000 or rp.stat().st_size > 2_000_000:
                continue
            ltxt=safe_read(lp).splitlines(True); rtxt=safe_read(rp).splitlines(True)
            out.writelines(difflib.unified_diff(ltxt, rtxt, fromfile='remote/'+rel, tofile='official/'+rel))
            out.write('\n')
    readme=safe_read(snap/'README.md')
    local_readme=safe_read(ego/'README.md')
    final_mentions={
        'official_final_eval_flag': '--final_eval' in readme,
        'official_309_tasks': bool(re.search(r'309\s+tasks|309', readme, re.I)),
        'official_5_files': all(x in readme for x in ['kitchen.json','order.json','restaurant.json','retail.json']),
        'official_technical_report_pdf': bool(re.search(r'technical report|pdf', readme, re.I)),
        'official_no_direct_final_json': bool(re.search(r'not.*directly.*read|directly.*read.*scenario|cheat', readme, re.I|re.S)),
        'local_final_eval_flag': '--final_eval' in local_readme,
        'local_309_tasks': bool(re.search(r'309\s+tasks|309', local_readme, re.I)),
        'local_5_files': all(x in local_readme for x in ['kitchen.json','order.json','restaurant.json','retail.json']),
        'local_technical_report_pdf': bool(re.search(r'technical report|pdf', local_readme, re.I)),
        'local_no_direct_final_json': bool(re.search(r'not.*directly.*read|directly.*read.*scenario|cheat', local_readme, re.I|re.S)),
    }
    tools_checks={}
    for rel in ['tools/order.py','tools/restaurant.py','tools/retail.py']:
        tools_checks[rel]=contains(snap/rel, {
            'restaurant_key_lookup':'restaurant.*key|key.*restaurant|restaurant_name',
            'selected_steaks_alias':'Selected Steaks|Steaks',
            'dish_name':'dish_name',
            'set_meal':'set_meal',
            'sauvignon_blanc':'Sauvignon Blanc|Sauvignon',
        })
    final_files=[]
    fd=ego/'scenarios'/'final'
    if fd.exists():
        final_files=[str(p.relative_to(fd)) for p in sorted(fd.glob('*.json'))]
    best=get_best(codex)
    latest=newest_reports(codex)
    # README final relevant excerpts, short paraphrase by regex line captures
    final_lines=[]
    for line in readme.splitlines():
        if any(k in line.lower() for k in ['final', 'submission', '309', 'technical report', '--final_eval', 'kitchen.json', 'order.json', 'restaurant.json', 'retail.json', 'cheat', 'directly']):
            s=line.strip()
            if s and len(final_lines)<80:
                final_lines.append(s)
    # Reports
    now=datetime.now().isoformat(timespec='seconds')
    lines=[]
    lines += [f'# Official Update Audit {args.timestamp}', '', f'- Generated: {now}', f'- Official commit: `{commit}`', f'- Remote EgoBench: `{ego}`', f'- Official snapshot: `{snap}`', f'- Compare CSV: `{csvp}`', f'- Diff patch preview: `{diffp}`', '']
    lines += ['## Summary', '', f'- Compared files: {len(rows)}', f'- Different files: {len(changed)}', f'- Missing locally: {len(missing_local)}', f'- Extra locally: {len(extra_local)}', '']
    lines += ['## Important Differences', '']
    for rel in changed[:80]: lines.append(f'- different: `{rel}`')
    for rel in missing_local[:80]: lines.append(f'- missing local: `{rel}`')
    for rel in extra_local[:40]: lines.append(f'- extra local: `{rel}`')
    if not changed and not missing_local:
        lines.append('- No official tracked-file update required for the compared set.')
    lines += ['', '## Final Guide Signals', '']
    for k,v in final_mentions.items(): lines.append(f'- {k}: `{v}`')
    lines += ['', '## Issue/PR Fix Heuristics From Official Snapshot', '']
    for rel,checks in tools_checks.items():
        lines.append(f'### {rel}')
        for k,v in checks.items(): lines.append(f'- {k}: `{v}`')
    lines += ['', '## Current Remote Final Files', '']
    if final_files:
        for f in final_files: lines.append(f'- `{f}`')
    else:
        lines.append('- No `scenarios/final/*.json` files found locally.')
    lines += ['', '## Recommendation', '']
    if 'README.md' in changed or missing_local:
        lines.append('- Update local `README.md` documentation copy so final-stage guidance is visible on the remote machine.')
    if any(x.startswith('tools/') or x.startswith('analysis_scripts/') or x.startswith('run/') or x=='run_all_scenarios.sh' for x in changed+missing_local):
        lines.append('- Review changed executable/evaluator/tool files before applying; official code changed in behavioral areas.')
    else:
        lines.append('- No behavior-critical official file difference detected in the tracked executable/tool/evaluator set.')
    lines.append('- Do not auto-submit final. Keep final packaging to the official 5 artifacts plus report PDF.')
    Path(args.report).write_text('\n'.join(lines)+'\n', encoding='utf-8')
    fgl=[]
    fgl += [f'# Final Stage Submission Guide {args.timestamp}', '', f'- Official commit: `{commit}`', f'- Source README: official GitHub snapshot at `{snap}`', '']
    fgl += ['## What Changed / What Matters', '', '- Final evaluation is run with `--final_eval`.', '- Final stage consists of 309 tasks across the held-out scenarios now present in `scenarios/final`.', '- Submission package should contain scenario-level JSON outputs: `kitchen.json`, `order.json`, `restaurant.json`, `retail.json`.', '- Submission also requires a technical report PDF.', '- The service agent must not directly inspect final scenario JSON contents; final scenario JSON is for the simulated user/evaluator pipeline, not for planner shortcuts.', '- We must not auto-submit. Generate package and report only.', '']
    fgl += ['## Remote Compliance Impact', '', '- Current wrapper/runner must be checked so visual priors and task metadata do not read `scenarios/final/*.json` directly in final mode.', '- Existing dev wrappers can use dev GT for diagnostics/training only; final runs must disable dev-GT-derived hints.', '- Final packer should emit exactly the four JSON files plus technical report PDF in the archive root.', '- Model/API usage must be disclosed in the technical report and internal draft report.', '']
    fgl += ['## Relevant README Lines Captured For Local Review', '']
    for s in final_lines:
        fgl.append(f'- {s[:240]}')
    Path(args.final_report).write_text('\n'.join(fgl)+'\n', encoding='utf-8')
    top=[]
    top += [f'# Top1 Readiness Analysis {args.timestamp}', '', f'- Official commit audited: `{commit}`', f'- Current best state file: `{codex / "state" / "best_track2_api_version.json"}`', '']
    top += ['## Current Best', '']
    if best:
        for k in ['version','run_id','joint_success','result_success','tool_success','micro_tool_accuracy','avg_tool_calls','model','endpoint','external_api_used','note']:
            if k in best: top.append(f'- {k}: `{best[k]}`')
    else:
        top.append('- No best state found.')
    top += ['', '## Verdict', '', '- Not top1-ready yet.', '- The current best 4-task gate has only 50% joint success. Since final has 309 held-out tasks, this is far below a plausible top1 trajectory unless order/kitchen generalization improves substantially.', '- The strongest parts are API connectivity, JSON/schema guards, duplicate mutation defense, and retail/restaurant preservation.', '- The biggest top1 blockers are order process-tool coverage and kitchen branch/tool-loop control. Result-only success is not enough because Track2 scores joint success over both tool process and final DB state.', '']
    top += ['## Fastest Path Toward Competitiveness', '', '- Keep `V6_1_3_gpt55_guarded_endpoint` as the stable base.', '- Apply only targeted order/kitchen patches; avoid broad V7-style behavior changes that degraded result success.', '- For order, align trajectory shape: pin restaurant, canonicalize dish/set meal/category, use expected aggregate compute tool, and repair missing process tools before final message.', '- For kitchen, allow required stock/location queries while preventing recipe over-scan; compute nutrition only from confirmed items.', '- Run official updated evaluator after syncing docs/tool/eval deltas, then expand to 10-20 easy tasks per scenario before any final run.', '- Final packaging must follow the new 5-artifact format and avoid direct final JSON reads.', '']
    top += ['## Latest Related Reports', '']
    for p in latest: top.append(f'- `{p}`')
    Path(args.top1_report).write_text('\n'.join(top)+'\n', encoding='utf-8')
    # update README_STATUS append
    status=codex/'README_STATUS.md'
    append='\n'.join(['', f'## Official Update Audit {args.timestamp}', '', f'- Official commit: `{commit}`', f'- Audit report: `{args.report}`', f'- Final guide report: `{args.final_report}`', f'- Top1 readiness: `{args.top1_report}`', f'- Current verdict: not top1-ready; best 4-task joint remains {best.get("joint_success", "unknown")}.', '- Final submission guide: final uses `--final_eval`; package four scenario JSON files plus technical report PDF; do not auto-submit; service agent must not directly read final scenario JSON.', ''])
    with status.open('a', encoding='utf-8') as f: f.write(append+'\n')
    print(json.dumps({'report':args.report,'final_report':args.final_report,'top1_report':args.top1_report,'csv':str(csvp),'diff':str(diffp),'changed':changed,'missing_local':missing_local,'extra_local':extra_local,'commit':commit}, ensure_ascii=False, indent=2))
if __name__=='__main__': main()
