#!/usr/bin/env bash
set -euo pipefail
cd /home/data-gxu/acm/egolink2026-main/code/track2/codex
ts=$(date +%Y%m%d_%H%M%S)
backup="backups/V11_preflight_v10_backup_${ts}"
mkdir -p "$backup"
cp -a memory_bank_v10 "$backup/"
mkdir -p "$backup/reports" "$backup/state" "$backup/submissions" "$backup/results"
cp -a reports/V10_* "$backup/reports/" 2>/dev/null || true
cp -a reports/FINAL_SUBMISSION_README_* "$backup/reports/" 2>/dev/null || true
cp -a state/*v10* "$backup/state/" 2>/dev/null || true
cp -a state/best_track2_api_version.json "$backup/state/" 2>/dev/null || true
cp -a submissions/V10_full_memory_final_candidate_draft_track2.zip "$backup/submissions/"
cp -a /home/data-gxu/acm/egolink2026-main/code/track2/EgoBench/results/V10_full_memory_final_candidate_draft "$backup/results/"
find "$backup" -type f | sort > "$backup/MANIFEST.txt"
sha256sum submissions/V10_full_memory_final_candidate_draft_track2.zip > "$backup/V10_ZIP_SHA256.txt"
cat > "reports/V11_BACKUP_STATE_${ts}.md" <<EOF
# V11 Preflight Backup ${ts}

- backup_dir: \`/home/data-gxu/acm/egolink2026-main/code/track2/codex/${backup}\`
- V10 zip preserved: yes
- V10 result copy: yes
- memory_bank_v10 copy: yes
- reports/state copy: yes
EOF
printf '%s\n' "$backup"
