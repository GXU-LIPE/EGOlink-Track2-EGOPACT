#!/usr/bin/env bash
set -euo pipefail

CODEX=/home/data-gxu/acm/egolink2026-main/code/track2/codex
CODE1=/home/data-gxu/acm/egolink2026-main/code1

cd "$CODE1"
source scripts/env.sh
cd "$CODEX"

export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export PYTHONIOENCODING=utf-8
export PYTHONPATH="$CODE1:$CODEX:$CODEX/scripts:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# GPT-5.5 cleaning uses ai-pixel endpoint from state/.openai_env. Keep keys out
# of logs and avoid proxy variables that break direct endpoint access here.
if [[ -f "$CODEX/state/.openai_env" ]]; then
  # shellcheck disable=SC1091
  source "$CODEX/state/.openai_env"
fi
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY="ai-pixel.online,cf.ai-pixel.online,localhost,127.0.0.1,::1"
export no_proxy="$NO_PROXY"

TS="$(date +%Y%m%d_%H%M%S)"
echo "[stage] gpu snapshot"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader

echo "[stage] generate real Qwen3-VL grounding cards for val41"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}" python3 scripts/build_v12_val41_qwen3vl_grounding.py \
  --materialized_dir "$CODEX/state/materialized_splits/validation_A_limit30" \
  --out_dir "$CODEX/visual_cache_v12/qwen3vl_grounding" \
  --frame_count 8 \
  --max_new_tokens 768 \
  --clear_out_dir \
  --require_qwen3vl

echo "[stage] strict GPT-5.5 cleanup for malformed/missing top_k"
python3 scripts/strict_clean_v12_val41_qwen3vl_cards.py \
  --card_dir "$CODEX/visual_cache_v12/qwen3vl_grounding" \
  --use_gpt55

echo "[stage] audit"
python3 scripts/audit_v12_val41_qwen3vl_cards.py

cat > "$CODEX/state/latest_v12_val41_qwen3vl_grounding_cards.json" <<JSON
{
  "updated_at": "$(date +%Y-%m-%dT%H:%M:%S%z)",
  "stage": "validation_A_medium_grounding_cards_only",
  "card_dir": "visual_cache_v12/qwen3vl_grounding",
  "task_count": 41,
  "final_run": false,
  "launcher_timestamp": "$TS"
}
JSON
echo "[done] generated val41 Qwen3-VL grounding cards"
