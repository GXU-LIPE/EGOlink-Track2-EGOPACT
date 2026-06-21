#!/usr/bin/env bash
set -euo pipefail
CODEX=/home/data-gxu/acm/egolink2026-main/code/track2/codex
TS=$(date +%Y%m%d_%H%M%S)
BACKUP="$CODEX/backups/v8_preflight_$TS"
mkdir -p "$BACKUP"
files=(
  state/best_track2_api_version.json
  README_STATUS.md
  runners/track2_multi_agent_plus.py
  runners/run_all_scenarios_plus.sh
  scripts/track2_pack_submission.py
  wrappers/egobench_agent_plus/planner.py
  wrappers/egobench_agent_plus/prompt_builder.py
  wrappers/egobench_agent_plus/db_guard.py
  wrappers/egobench_agent_plus/tool_validator.py
  wrappers/egobench_agent_plus/service_agent_wrapper.py
  wrappers/egobench_agent_plus/process_coverage_verifier.py
  wrappers/egobench_agent_plus/human_prior_controller.py
)
for rel in "${files[@]}"; do
  if [[ -f "$CODEX/$rel" ]]; then
    mkdir -p "$BACKUP/$(dirname "$rel")"
    cp -p "$CODEX/$rel" "$BACKUP/$rel"
  fi
done
{
  echo "# V8 Preflight Backup $TS"
  echo
  echo "- backup: $BACKUP"
  echo "- no final submission: true"
  echo
  echo "## Files"
  find "$BACKUP" -type f -printf '%P\n' | sort | while read -r f; do
    sha=$(sha256sum "$BACKUP/$f" | awk '{print $1}')
    echo "- $f sha256=$sha"
  done
} > "$CODEX/reports/V8_PREFLIGHT_BACKUP_$TS.md"
echo "$TS"
echo "$BACKUP"
echo "$CODEX/reports/V8_PREFLIGHT_BACKUP_$TS.md"
