#!/usr/bin/env bash
set -euo pipefail

CODEX=/home/data-gxu/acm/egolink2026-main/code/track2/codex
EGO=/home/data-gxu/acm/egolink2026-main/code/track2/EgoBench
cd "$CODEX"

mkdir -p logs reports state analysis

python3 -m py_compile \
  wrappers/egobench_agent_plus/direct_api.py \
  wrappers/egobench_agent_plus/openai_gpt55_adapter.py \
  runners/track2_multi_agent_plus.py

# The key is injected by the caller into this process environment only.
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY missing" >&2
  exit 2
fi

export TRACK2_OPENAI_MODEL="${TRACK2_OPENAI_MODEL:-gpt-5.5}"
export TRACK2_OPENAI_USER_MODEL="${TRACK2_OPENAI_USER_MODEL:-gpt-5.5}"
export TRACK2_USER_USE_OPENAI=1
export TRACK2_OPENAI_REASONING_EFFORT="${TRACK2_OPENAI_REASONING_EFFORT:-medium}"
export TRACK2_OPENAI_TEXT_VERBOSITY="${TRACK2_OPENAI_TEXT_VERBOSITY:-low}"
export TRACK2_OPENAI_MAX_OUTPUT_TOKENS="${TRACK2_OPENAI_MAX_OUTPUT_TOKENS:-2048}"
export TRACK2_OPENAI_TIMEOUT="${TRACK2_OPENAI_TIMEOUT:-180}"
export TRACK2_OPENAI_MAX_RETRIES="${TRACK2_OPENAI_MAX_RETRIES:-1}"
export TRACK2_MAX_TURNS="${TRACK2_MAX_TURNS:-8}"
export TRACK2_SKIP_VISUAL_STATE="${TRACK2_SKIP_VISUAL_STATE:-1}"

bash scripts/run_gpt55_v6_gate.sh

latest_gate=$(find reports -maxdepth 1 -type f -name '02_gpt55_gate_summary_*' -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
echo "LATEST_GATE=$latest_gate"
sed -n '1,220p' "$latest_gate"
