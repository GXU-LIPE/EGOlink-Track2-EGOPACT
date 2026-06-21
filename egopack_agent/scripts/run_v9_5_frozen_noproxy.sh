#!/usr/bin/env bash
set -euo pipefail
cd /home/data-gxu/acm/egolink2026-main/code/track2/codex
source state/.openai_env
unset HTTPS_PROXY HTTP_PROXY ALL_PROXY https_proxy http_proxy all_proxy
export NO_PROXY="ai-pixel.online,cf.ai-pixel.online,localhost,127.0.0.1,::1"
export no_proxy="$NO_PROXY"
export TRACK2_OPENAI_BASE_URL="${TRACK2_OPENAI_BASE_URL:-https://ai-pixel.online/v1}"
export SERVICE_MODEL_API_BASE="$TRACK2_OPENAI_BASE_URL"
export TRACK2_API_MAX_RETRIES=1
export TRACK2_CONNECT_TIMEOUT=10
export TRACK2_READ_TIMEOUT=240
export TRACK2_V8_TASK_TIMEOUT=1800
export TRACK2_ENABLE_MULTICANDIDATE=1
export TRACK2_ENABLE_MEMORY_RETRIEVAL=1
export TRACK2_ENABLE_VISUAL_GROUNDING_RESOLVER=1
export TRACK2_ENABLE_RETAIL_NARROWER=1
export TRACK2_ENABLE_ORDER_PROCESS_MEMORY=1
export TRACK2_ENABLE_DEEPSEEK_CROSSCHECK=0
export TRACK2_USE_DEEPSEEK_CROSSCHECK=0
RUN_ID="V9_5_memory_deepseek_rerank_validation_A_medium_$(date +%Y%m%d_%H%M)_frozen_noproxy"
python3 scripts/run_v8_validation.py --stage validation_A_medium --version V9_5_memory_deepseek_rerank --run-id "${RUN_ID}_dryrun" --model gpt-5.5 --dry-run
python3 scripts/run_v8_validation.py --stage validation_A_medium --version V9_5_memory_deepseek_rerank --run-id "$RUN_ID" --model gpt-5.5
bash scripts/check_v8_status.sh "$RUN_ID" V9_5_memory_deepseek_rerank || true
python3 - <<'PY'
import json, pathlib, time
codex=pathlib.Path('/home/data-gxu/acm/egolink2026-main/code/track2/codex')
version='V9_5_memory_deepseek_rerank'
runs=sorted((codex/'runs'/version).glob('V9_5_memory_deepseek_rerank_validation_A_medium_*_frozen_noproxy'), key=lambda p:p.stat().st_mtime)
run=runs[-1]
eval_path=run/'eval_summary.json'
data=json.loads(eval_path.read_text())
s=data.get('summary',{})
ts=time.strftime('%Y%m%d_%H%M%S')
report=codex/'reports'/f'V9_VALIDATION_A_MEDIUM_RERUN_{ts}.md'
lines=[f'# V9 Validation A Medium Rerun {ts}','',f'- run_id: `{run.name}`','- version: `V9_5_memory_deepseek_rerank`','- split: materialized validation_A_limit30 frozen from V9_4_5 41-task subset','- no_proxy_api: true','- deepseek_crosscheck_online: false (key not configured / disabled)','- protected_best_updated: false','- final_submission: not submitted','', '## Summary','', f"- valid: {s.get('valid',0)}", f"- joint: {s.get('joint',0):.4f}", f"- result: {s.get('result',0):.4f}", f"- tool: {s.get('tool',0):.4f}", f"- micro: {s.get('micro',0):.4f}", f"- calls: {s.get('correct_calls',0)}/{s.get('gt_calls',0)} gt, interaction_calls={s.get('interaction_calls',0)}",'', '## Decision','', '- Do not run final unless this and validation_B_holdout exceed protected V6.']
report.write_text('\n'.join(lines)+'\n')
for prefix in ['V9_5_DEEPSEEK_RERANKER','V9_NEXT_TOP1_READINESS']:
    p=codex/'reports'/f'{prefix}_{ts}.md'
    extra='DeepSeek online crosscheck was disabled; this run tests deterministic V9.5 reranker/split fixes only.' if prefix=='V9_5_DEEPSEEK_RERANKER' else 'Protected best remains V6_1_3_gpt55_guarded_endpoint unless A_medium and validation_B_holdout both beat it.'
    p.write_text('\n'.join([f'# {prefix} {ts}','',f'- run_id: `{run.name}`',f'- valid: {s.get("valid",0)}',f'- joint: {s.get("joint",0):.4f}',f'- result: {s.get("result",0):.4f}',f'- tool: {s.get("tool",0):.4f}',f'- micro: {s.get("micro",0):.4f}',f'- note: {extra}','- final_submission: not submitted','']) )
print(json.dumps({'run_id':run.name,'summary':s,'report':str(report)}, ensure_ascii=False, indent=2))
PY
