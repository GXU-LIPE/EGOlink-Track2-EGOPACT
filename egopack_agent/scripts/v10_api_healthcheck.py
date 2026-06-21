#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import json, os, time
from pathlib import Path
import requests

CODEX=Path(os.environ.get('CODEX_ROOT','/home/data-gxu/acm/egolink2026-main/code/track2/codex'))

def load_env(path):
    if not path.exists(): return
    for line in path.read_text().splitlines():
        line=line.strip()
        if not line.startswith('export ') or '=' not in line: continue
        k,v=line[len('export '):].split('=',1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

def check_openai():
    load_env(CODEX/'state'/'.openai_env')
    key=os.environ.get('OPENAI_API_KEY') or os.environ.get('SERVICE_MODEL_API_KEY')
    base=os.environ.get('TRACK2_OPENAI_BASE_URL') or os.environ.get('SERVICE_MODEL_API_BASE') or 'https://ai-pixel.online/v1'
    model=os.environ.get('TRACK2_OPENAI_MODEL') or os.environ.get('SERVICE_MODEL_NAME') or 'gpt-5.5'
    out={'base_url':base,'model':model,'key_present':bool(key),'ok':False}
    if not key:
        out['error_class']='MissingKey'; return out
    try:
        r=requests.post(base.rstrip()+'/chat/completions',headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'},json={'model':model,'messages':[{'role':'user','content':'Reply OK only.'}],'temperature':0,'max_tokens':8,'stream':False},timeout=(10,60),proxies={})
        out['http_status']=r.status_code
        if r.ok:
            content=((r.json().get('choices') or [{}])[0].get('message') or {}).get('content','')
            out['ok']=bool(content.strip())
            out['response_preview']=content.strip()[:20]
        else:
            out['error_preview']=r.text[:200]
    except Exception as e:
        out['error_class']=type(e).__name__; out['error_preview']=str(e)[:200]
    return out

def check_deepseek():
    load_env(CODEX/'state'/'.deepseek_env')
    key=os.environ.get('DEEPSEEK_API_KEY') or os.environ.get('TRACK2_DEEPSEEK_API_KEY')
    base=os.environ.get('DEEPSEEK_API_BASE') or os.environ.get('TRACK2_DEEPSEEK_API_BASE') or 'https://api.deepseek.com/v1'
    model=os.environ.get('TRACK2_DEEPSEEK_CROSSCHECK_MODEL','deepseek-chat')
    out={'base_url':base,'model':model,'key_present':bool(key),'ok':False,'enabled_for_v10':False}
    if not key:
        out['disabled_reason']='missing_key'; return out
    try:
        r=requests.post(base.rstrip()+'/chat/completions',headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'},json={'model':model,'messages':[{'role':'user','content':'Return JSON {"risk":"low"}'}],'temperature':0,'max_tokens':40,'stream':False},timeout=(8,40),proxies={})
        out['http_status']=r.status_code; out['ok']=r.ok; out['enabled_for_v10']=r.ok
        if not r.ok: out['error_preview']=r.text[:200]
    except Exception as e:
        out['error_class']=type(e).__name__; out['error_preview']=str(e)[:200]
    return out

def main():
    ts=time.strftime('%Y%m%d_%H%M%S')
    os.environ.pop('HTTPS_PROXY',None); os.environ.pop('HTTP_PROXY',None); os.environ.pop('ALL_PROXY',None)
    os.environ.pop('https_proxy',None); os.environ.pop('http_proxy',None); os.environ.pop('all_proxy',None)
    os.environ['NO_PROXY']='ai-pixel.online,cf.ai-pixel.online,localhost,127.0.0.1,::1'
    health={'generated_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'gpt55':check_openai(),'deepseek':check_deepseek(),'no_proxy':os.environ.get('NO_PROXY')}
    state=CODEX/'state'/f'v10_api_health_{ts}.json'; state.parent.mkdir(exist_ok=True)
    state.write_text(json.dumps(health,ensure_ascii=False,indent=2),encoding='utf-8')
    report=CODEX/'reports'/f'V10_API_HEALTHCHECK_{ts}.md'
    lines=[f'# V10 API Healthcheck {ts}','',f"- GPT-5.5 key_present: {health['gpt55'].get('key_present')}",f"- GPT-5.5 base_url: `{health['gpt55'].get('base_url')}`",f"- GPT-5.5 model: `{health['gpt55'].get('model')}`",f"- GPT-5.5 ok: {health['gpt55'].get('ok')}",f"- DeepSeek key_present: {health['deepseek'].get('key_present')}",f"- DeepSeek ok: {health['deepseek'].get('ok')}",f"- DeepSeek enabled_for_v10: {health['deepseek'].get('enabled_for_v10')}",'- API keys redacted: yes','']
    if not health['gpt55'].get('ok'):
        block=CODEX/'reports'/f'NEED_HUMAN_ATTENTION_GPT55_API_{ts}.md'
        block.write_text('\n'.join(lines+[f"GPT-5.5 healthcheck failed: {health['gpt55'].get('error_class','http')} {health['gpt55'].get('error_preview','')}\n"]),encoding='utf-8')
    report.write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'report':str(report),'state':str(state),'health':health},ensure_ascii=False,indent=2))
    if not health['gpt55'].get('ok'):
        raise SystemExit(2)
if __name__=='__main__': main()
