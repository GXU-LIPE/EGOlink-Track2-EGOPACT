#!/usr/bin/env bash
set -euo pipefail

CODEX=/home/data-gxu/acm/egolink2026-main/code/track2/codex
PYTHON_BIN="${TRACK2_PYTHON:-python3}"
STAMP="$(date +%Y%m%d_%H%M%S)"

cd "$CODEX"
mkdir -p reports state logs

{
  echo "# Track2 API Verification"
  echo
  echo "- generated_at: $(date +%Y-%m-%dT%H:%M:%S%z)"
  echo "- secrets_logged: false"
  echo
} > "reports/TRACK2_API_VERIFY_${STAMP}.md"

# GPT-5.5 proven path: old endpoint gate explicitly unsets stale proxy vars.
set +e
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
GPT_OUT="$($PYTHON_BIN scripts/track2_endpoint_probe.py 2>&1)"
GPT_RC=$?
set -e
printf '%s\n' "$GPT_OUT" > "logs/track2_gpt55_verify_${STAMP}.log"

{
  echo "## GPT-5.5"
  echo
  echo "- route: scripts/track2_endpoint_probe.py"
  echo "- proxy: unset"
  echo "- exit_code: $GPT_RC"
  echo "- ok: $([ "$GPT_RC" -eq 0 ] && echo true || echo false)"
  echo
  echo '```json'
  printf '%s\n' "$GPT_OUT" | "$PYTHON_BIN" -c 'import sys,re; print(re.sub(r"sk-[A-Za-z0-9_-]+","sk-[REDACTED]",sys.stdin.read())[:4000])'
  echo '```'
  echo
} >> "reports/TRACK2_API_VERIFY_${STAMP}.md"

# DeepSeek path 1: old successful config from state/secrets.env.
set +a
if [ -f state/secrets.env ]; then
  set -a
  # shellcheck disable=SC1091
  . state/secrets.env
  set +a
fi

set +e
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
DEEP_OUT_DIRECT="$($PYTHON_BIN scripts/track2_deepseek_api_check.py --base-url "${SERVICE_API_BASE_URL:-https://api.deepseek.com/v1}" --models deepseek-v4-pro deepseek-v4-flash deepseek-chat --connect-timeout 8 --read-timeout 45 2>&1)"
DEEP_RC_DIRECT=$?
set -e
printf '%s\n' "$DEEP_OUT_DIRECT" > "logs/track2_deepseek_verify_direct_${STAMP}.log"

{
  echo "## DeepSeek Direct"
  echo
  echo "- route: scripts/track2_deepseek_api_check.py"
  echo "- proxy: unset"
  echo "- exit_code: $DEEP_RC_DIRECT"
  echo "- ok: $([ "$DEEP_RC_DIRECT" -eq 0 ] && echo true || echo false)"
  echo
  echo '```json'
  printf '%s\n' "$DEEP_OUT_DIRECT" | "$PYTHON_BIN" -c 'import sys,re; print(re.sub(r"sk-[A-Za-z0-9_-]+","sk-[REDACTED]",sys.stdin.read())[:4000])'
  echo '```'
  echo
} >> "reports/TRACK2_API_VERIFY_${STAMP}.md"

echo "reports/TRACK2_API_VERIFY_${STAMP}.md"
