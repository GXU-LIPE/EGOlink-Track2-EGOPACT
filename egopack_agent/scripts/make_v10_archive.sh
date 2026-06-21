#!/usr/bin/env bash
set -euo pipefail
cd /home/data-gxu/acm/egolink2026-main/code/track2/codex
out="/home/data-gxu/acm/egolink2026-main/code/track2/codex/reports/V10_final_candidate_archive_20260619_0012.tar.gz"
rm -f "$out"
tar -czf "$out" \
  reports/V10_FINAL_CANDIDATE_README_20260619_001219.md \
  reports/V10_FINAL_FULL_SANITY_20260619_001219.md \
  reports/V10_SUBMISSION_PACKAGE_DRAFT_20260619_001219.md \
  reports/V10_TECHNICAL_REPORT_DRAFT_V10_full_memory_final_candidate_draft_20260618_1940.md \
  reports/V10_full_memory_final_candidate_draft.pdf \
  reports/V10_RESULT_RISK_AUDIT_20260619_0012.jsonl \
  reports/V10_API_HEALTHCHECK_20260618_171510.md \
  reports/V10_FULL_MEMORY_BANK_BUILD_20260618_171530.md \
  reports/V10_MEMORY_COVERAGE_AUDIT_20260618_171530.md \
  reports/V10_A_MEDIUM_SANITY_20260618_183911.md \
  reports/V10_FINAL_SMOKE_SANITY_20260618_193820.md \
  state/v10_final_candidate_package.json \
  submissions/V10_full_memory_final_candidate_draft_track2.zip
ls -lh "$out"
sha256sum "$out"
