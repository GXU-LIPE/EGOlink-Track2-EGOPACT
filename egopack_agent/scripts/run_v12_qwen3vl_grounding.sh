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
export PYTHONPATH="$CODE1:$CODEX:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
CUDA_VISIBLE_DEVICES=0,1,2 python3 scripts/build_v12_qwen3vl_grounding.py \
  --specs retail9 restaurant4 order1 kitchen2 \
  --frame_count 8 \
  --max_new_tokens 512 \
  --skip_existing
