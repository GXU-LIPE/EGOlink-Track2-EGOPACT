#!/usr/bin/env bash
set -euo pipefail
cd /home/data-gxu/acm/egolink2026-main/code/track2/codex
TS=$(date +%Y%m%d_%H%M%S)
BK="backups/V10_full_memory_final_candidate_draft_${TS}"
mkdir -p "$BK"
cp -a runners/track2_multi_agent_plus.py "$BK/"
mkdir -p "$BK/wrappers" "$BK/scripts" "$BK/state"
cp -a wrappers/egobench_agent_plus/memory_retriever.py wrappers/egobench_agent_plus/prompt_builder.py wrappers/egobench_agent_plus/service_agent_wrapper.py wrappers/egobench_agent_plus/v9_candidate_builder.py wrappers/egobench_agent_plus/v9_multicandidate_reranker.py "$BK/wrappers/"
cp -a scripts/run_v8_validation.py scripts/track2_pack_submission.py "$BK/scripts/"
cp -a state/best_track2_api_version.json state/candidate_track2_api_version.json "$BK/state/" 2>/dev/null || true
sha256sum "$BK"/*.py "$BK"/wrappers/*.py "$BK"/scripts/*.py "$BK"/state/*.json 2>/dev/null > "$BK/SHA256SUMS.txt" || true
REPORT="reports/V10_BACKUP_STATE_${TS}.md"
{
 echo "# V10 Backup State ${TS}"
 echo
 echo "- backup_dir: \`$BK\`"
 echo "- includes_key_files: no"
 echo "- protected_best_changed: no"
 echo
 echo "## Files"
 find "$BK" -maxdepth 3 -type f | sort | sed 's#^#- `#;s#$#`#'
} > "$REPORT"
echo "$REPORT"
