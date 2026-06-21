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
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

TS="$(date +%Y%m%d_%H%M%S)"
RUN_ID="V12_official_style_qwen3vl_memory_val41_all_qwen3vl_${TS}"
LOG_DIR="$CODEX/logs"
mkdir -p "$LOG_DIR" "$CODEX/reports"

echo "[stage] build qwen3vl grounding cards"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}" python3 scripts/build_v12_val41_qwen3vl_grounding.py \
  --materialized_dir "$CODEX/state/materialized_splits/validation_A_limit30" \
  --out_dir "$CODEX/visual_cache_v12/qwen3vl_grounding" \
  --frame_count 8 \
  --max_new_tokens 768 \
  --clear_out_dir

echo "[stage] clean cards with GPT-5.5 if needed"
python3 scripts/clean_v12_val41_qwen3vl_cards.py \
  --card_dir "$CODEX/visual_cache_v12/qwen3vl_grounding" \
  --use_gpt55

echo "[stage] audit cards"
python3 scripts/audit_v12_val41_qwen3vl_cards.py

echo "[stage] run val41 with all qwen3vl cards"
TRACK2_RUN_ID="$RUN_ID" bash scripts/run_v12_val41.sh

echo "[stage] final audit after run"
python3 scripts/audit_v12_val41_qwen3vl_cards.py

cat > "$CODEX/state/latest_v12_val41_all_qwen3vl.json" <<JSON
{
  "updated_at": "$(date +%Y-%m-%dT%H:%M:%S%z)",
  "run_id": "$RUN_ID",
  "version": "V12_official_style_qwen3vl_memory",
  "stage": "validation_A_medium",
  "qwen3vl_cards": "visual_cache_v12/qwen3vl_grounding",
  "final_run": false
}
JSON
echo "[done] $RUN_ID"
