#!/usr/bin/env bash
set -euo pipefail
CODEX=/home/data-gxu/acm/egolink2026-main/code/track2/codex
MODEL_PATH="${SERVICE_MODEL_PATH:-/home/data-gxu/acm/egolink2026-main/code1/models/Qwen/Qwen2___5-VL-32B-Instruct}"
PYTHON_BIN="${TRACK2_QWEN_PYTHON:-python3}"
export TRACK2_FINAL_COMPLIANT=1
export SERVICE_MODEL_BACKEND=local_qwen_vllm
export SERVICE_MODEL_NAME=Qwen2.5-VL-32B-Instruct
export SERVICE_MODEL_PATH="$MODEL_PATH"
export SERVICE_MODEL_API_BASE="${SERVICE_MODEL_API_BASE:-http://127.0.0.1:8000/v1}"
cd "$CODEX"
exec "$PYTHON_BIN" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name Qwen2.5-VL-32B-Instruct \
  --tensor-parallel-size "${TRACK2_QWEN_TP:-3}" \
  --dtype bfloat16 \
  --max-model-len "${TRACK2_QWEN_MAX_LEN:-32768}" \
  --gpu-memory-utilization "${TRACK2_QWEN_GPU_MEM:-0.85}" \
  --port "${TRACK2_QWEN_PORT:-8000}"
