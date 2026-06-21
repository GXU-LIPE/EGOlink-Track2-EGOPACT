#!/usr/bin/env bash
set -euo pipefail

CODEX=/home/data-gxu/acm/egolink2026-main/code/track2/codex
CODE1=/home/data-gxu/acm/egolink2026-main/code1
OUT_DIR="$CODEX/visual_cache_v12/qwen3vl_grounding_all_dev_offline"

cd "$CODE1"
source scripts/env.sh
cd "$CODEX"

export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export PYTHONIOENCODING=utf-8
export PYTHONPATH="$CODE1:$CODEX:$CODEX/scripts:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

if [[ -f "$CODEX/state/.openai_env" ]]; then
  # shellcheck disable=SC1091
  source "$CODEX/state/.openai_env"
fi
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY="ai-pixel.online,cf.ai-pixel.online,localhost,127.0.0.1,::1"
export no_proxy="$NO_PROXY"

mkdir -p logs reports "$OUT_DIR"

echo "[stage] gpu snapshot"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true

echo "[stage] generate real Qwen3-VL grounding cards for all non-final dev/offline tasks"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}" python3 scripts/build_v12_all_qwen3vl_grounding_by_video.py \
  --out_dir "$OUT_DIR" \
  --require_qwen3vl \
  --frame_count 12 \
  --max_new_tokens 2048 \
  --clear_out_dir \
  --skip_existing_video

echo "[stage] strict GPT-5.5 cleanup for malformed/missing top_k"
python3 scripts/strict_clean_v12_qwen3vl_cards.py \
  --card_dir "$OUT_DIR" \
  --use_gpt55

echo "[stage] audit"
python3 scripts/audit_v12_qwen3vl_cards.py \
  --card_dir "$OUT_DIR" \
  --report_prefix V12_ALL_DEV_OFFLINE_QWEN3VL_GROUNDING_AUDIT

python3 - <<'PY'
import json, time
from pathlib import Path
root = Path('/home/data-gxu/acm/egolink2026-main/code/track2/codex')
reports = sorted((root/'reports').glob('V12_ALL_DEV_OFFLINE_QWEN3VL_GROUNDING_AUDIT_*.json'), key=lambda p: p.stat().st_mtime, reverse=True)
state = {
    'updated_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
    'card_dir': str(root/'visual_cache_v12/qwen3vl_grounding_all_dev_offline'),
    'latest_audit_json': str(reports[0]) if reports else '',
    'final_hidden_metadata_used': False,
}
(root/'state/latest_v12_all_dev_offline_qwen3vl_grounding_cards.json').write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
print('[done] generated all non-final dev/offline Qwen3-VL grounding cards')
PY
