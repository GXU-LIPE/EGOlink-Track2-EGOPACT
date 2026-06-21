#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import json, os, re, time
from pathlib import Path

CODEX=Path('/home/data-gxu/acm/egolink2026-main/code/track2/codex')
REPORTS=CODEX/'reports'
REPORTS.mkdir(parents=True, exist_ok=True)
ROOTS=[Path('/data/home-gxu/wjb19/egotrack1'),Path('/home/data-gxu/acm/egolink2026-main'),Path('/home/data-gxu/acm/egolink2026-main/code1')]
KEYS=re.compile(r'qwen3[-_ ]?vl|qwen3vl|qwen3_vl|qwen-3-vl|video.*mllm|mllm', re.I)
SCRIPT_EXT={'.py','.sh','.yaml','.yml','.json','.md','.txt'}
MODEL_HINTS=('checkpoint','ckpt','model','weights','safetensors','Qwen','qwen')

found=[]
for root in ROOTS:
    if not root.exists():
        found.append({'root':str(root),'status':'missing'})
        continue
    # Scoped walk, skip obvious heavy dirs but still record qwen-looking dirs.
    count=0
    for dirpath, dirnames, filenames in os.walk(root):
        p=Path(dirpath)
        rel_depth=len(p.relative_to(root).parts) if p!=root else 0
        # prune very deep generic dependency dirs
        dirnames[:] = [d for d in dirnames if d not in {'.git','__pycache__','node_modules','.cache','wandb','runs','logs'}]
        name_hit=bool(KEYS.search(str(p)))
        if name_hit:
            try:
                size=sum((p/f).stat().st_size for f in filenames[:200] if (p/f).is_file())
            except Exception:
                size=None
            found.append({'root':str(root),'type':'dir','path':str(p),'depth':rel_depth,'file_count':len(filenames),'sample_files':filenames[:20],'sample_size':size})
            count+=1
        for fn in filenames:
            fp=p/fn
            text=str(fp)
            if KEYS.search(text) or (fp.suffix in SCRIPT_EXT and any(k.lower() in fn.lower() for k in ['qwen','vl','video','infer','eval'])):
                try:
                    st=fp.stat()
                    if st.st_size > 5_000_000 and not KEYS.search(text):
                        continue
                    snippet=''
                    if fp.suffix in SCRIPT_EXT and st.st_size < 300_000:
                        data=fp.read_text(encoding='utf-8', errors='replace')[:5000]
                        if KEYS.search(data) or KEYS.search(text):
                            snippet='\n'.join(data.splitlines()[:30])
                        elif not any(k in fn.lower() for k in ['qwen','infer','eval','video']):
                            continue
                    found.append({'root':str(root),'type':'file','path':str(fp),'size':st.st_size,'snippet':snippet[:2000]})
                    count+=1
                except Exception as e:
                    found.append({'root':str(root),'type':'file','path':str(fp),'error':str(e)})
                    count+=1
        if count>250:
            found.append({'root':str(root),'status':'truncated','reason':'more than 250 matches'})
            break
        # avoid walking all of giant repo under irrelevant deep dirs if no path/name hints
        if rel_depth>7 and not name_hit:
            dirnames[:] = []

ts=time.strftime('%Y%m%d_%H%M%S')
out=REPORTS/f'V12_QWEN3VL_INVENTORY_{ts}.json'
out.write_text(json.dumps({'generated_at':ts,'roots':[str(r) for r in ROOTS],'matches':found},ensure_ascii=False,indent=2),encoding='utf-8')
md=REPORTS/f'V12_QWEN3VL_INVENTORY_{ts}.md'
lines=['# V12 Qwen3-VL Inventory','',f'- generated_at: `{ts}`','- purpose: locate Track1 Qwen3-VL video-MLLM inference resources for grounding teacher','']
valid=[x for x in found if x.get('type') in {'file','dir'}]
lines.append(f'- total_matches: `{len(valid)}`')
for root in ROOTS:
    lines.append(f'- root `{root}` exists: `{root.exists()}`')
lines.append('')
for item in valid[:80]:
    lines.append(f"- {item.get('type')}: `{item.get('path')}` size={item.get('size', item.get('sample_size',''))}")
    if item.get('sample_files'):
        lines.append(f"  sample_files: `{', '.join(map(str,item.get('sample_files')[:8]))}`")
    if item.get('snippet'):
        first=' '.join(item['snippet'].split())[:300]
        lines.append(f"  snippet: `{first}`")
if not valid:
    need=REPORTS/f'NEED_HUMAN_ATTENTION_QWEN3VL_{ts}.md'
    need.write_text('# NEED_HUMAN_ATTENTION_QWEN3VL\n\nNo usable Qwen3-VL/qwen3vl resource was found in the requested roots. V12 can continue without the video-MLLM teacher, but the Qwen3-VL grounding part is unavailable until a script/env/model path is supplied.\n',encoding='utf-8')
    lines.append('')
    lines.append(f'- blocker_report: `{need}`')
md.write_text('\n'.join(lines),encoding='utf-8')
print(out)
print(md)
