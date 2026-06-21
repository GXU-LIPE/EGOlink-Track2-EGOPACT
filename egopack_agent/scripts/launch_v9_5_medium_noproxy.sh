#!/usr/bin/env bash
set -euo pipefail
cd /home/data-gxu/acm/egolink2026-main/code/track2/codex

# The local proxy at 127.0.0.1:17897 intermittently stalls CONNECT for
# ai-pixel endpoints. This validation run intentionally bypasses it.
unset HTTPS_PROXY HTTP_PROXY ALL_PROXY https_proxy http_proxy all_proxy
export NO_PROXY="ai-pixel.online,cf.ai-pixel.online,localhost,127.0.0.1,::1"
export no_proxy="$NO_PROXY"
export TRACK2_CONNECT_TIMEOUT="${TRACK2_CONNECT_TIMEOUT:-20}"
export TRACK2_READ_TIMEOUT="${TRACK2_READ_TIMEOUT:-300}"
export TRACK2_API_MAX_RETRIES="${TRACK2_API_MAX_RETRIES:-2}"
export TRACK2_SERVICE_FALLBACK_MODELS=""

run_id="V9_5_memory_deepseek_rerank_validation_A_medium_$(date +%Y%m%d_%H%M)_noproxy"
log="logs/${run_id}.launcher.log"
mkdir -p logs
nohup python3 scripts/run_v8_validation.py \
  --stage validation_A_medium \
  --version V9_5_memory_deepseek_rerank \
  --run-id "$run_id" \
  --limit-per-scenario 30 \
  > "$log" 2>&1 &
echo "run_id=$run_id"
echo "pid=$!"
echo "log=$log"
