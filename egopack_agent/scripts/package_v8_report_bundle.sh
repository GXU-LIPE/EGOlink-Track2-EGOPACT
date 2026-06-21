#!/usr/bin/env bash
set -euo pipefail
cd /home/data-gxu/acm/egolink2026-main/code/track2/codex
bundle_dir="runs/report_bundles/v8_top1_readiness_20260617"
rm -rf "${bundle_dir}"
mkdir -p "${bundle_dir}/reports" "${bundle_dir}/state" "${bundle_dir}/scripts" "${bundle_dir}/analysis"

cp -f reports/V8_TOP1_READINESS_20260617_210513.md "${bundle_dir}/reports/" 2>/dev/null || true
cp -f reports/V8_SMOKE_SUMMARY_v8_smoke_20260617_1823_corrected.md "${bundle_dir}/reports/" 2>/dev/null || true
cp -f reports/V8_SMOKE_SUMMARY_v8_0_smoke_20260617_1832_corrected.md "${bundle_dir}/reports/" 2>/dev/null || true
cp -f reports/V8_SMOKE_SUMMARY_v8_1_smoke_20260617_continue.md "${bundle_dir}/reports/" 2>/dev/null || true
cp -f reports/V8_SMOKE_SUMMARY_v8_2_smoke_20260617_continue.md "${bundle_dir}/reports/" 2>/dev/null || true
cp -f reports/V8_VALIDATION_A_SMALL_v8_2_valA_small_20260617_continue.md "${bundle_dir}/reports/" 2>/dev/null || true
cp -f reports/V8_VALIDATION_A_SMALL_v8_0_valA_small_20260617_continue.md "${bundle_dir}/reports/" 2>/dev/null || true
cp -f reports/V8_DATASET_COUNT_AUDIT_20260617_180809.md "${bundle_dir}/reports/" 2>/dev/null || true
cp -f reports/FINAL_STAGE_SUBMISSION_GUIDE_20260617_170840.md "${bundle_dir}/reports/" 2>/dev/null || true
cp -f reports/TOP1_READINESS_ANALYSIS_20260617_170840.md "${bundle_dir}/reports/" 2>/dev/null || true

cp -f state/best_track2_api_version.json "${bundle_dir}/state/" 2>/dev/null || true
cp -f state/v8_top1_readiness_latest.json "${bundle_dir}/state/" 2>/dev/null || true
cp -f state/track2_data_split_latest.json "${bundle_dir}/state/" 2>/dev/null || true
cp -f analysis/dataset_task_counts_20260617_180809.csv "${bundle_dir}/analysis/" 2>/dev/null || true
cp -f analysis/dataset_task_splits_20260617_180809.csv "${bundle_dir}/analysis/" 2>/dev/null || true

cp -f scripts/run_v8_validation.py "${bundle_dir}/scripts/" 2>/dev/null || true
cp -f scripts/write_v8_top1_report.py "${bundle_dir}/scripts/" 2>/dev/null || true
cp -f README_STATUS.md "${bundle_dir}/" 2>/dev/null || true

cat > "${bundle_dir}/LOCAL_ARCHIVE_INDEX.md" <<'MD'
# Local Archive Index

- Project: VLN Track2
- Experiment / attempt: V8 top1 readiness long validation
- Completed: 2026-06-17 Asia/Shanghai
- Remote alias: ego-a100-old
- Remote source: `/home/data-gxu/acm/egolink2026-main/code/track2/codex`

## Key Result

V8 validation micro extraction was fixed. V8_2 kitchen helper reached 75% joint on the fixed 4-task smoke, but only 5% joint and 24.71% micro on frozen 20-task validation_A_small. Protected best remains V6_1_3_gpt55_guarded_endpoint. No final submission was made.

## Included Files

- V8 readiness and validation reports
- Dataset count/split audit
- Final stage submission guide from prior official sync
- Best/version state JSON
- V8 validation runner scripts

## Not Included

- Raw full interaction logs and result JSON trees, to keep the bundle compact.
- API keys and `.openai_env`.
- Datasets, videos, checkpoints, and visual cache.
MD

tar -czf runs/report_bundles/v8_top1_readiness_20260617.tar.gz -C runs/report_bundles v8_top1_readiness_20260617
sha256sum runs/report_bundles/v8_top1_readiness_20260617.tar.gz > runs/report_bundles/v8_top1_readiness_20260617.tar.gz.sha256
ls -lh runs/report_bundles/v8_top1_readiness_20260617.tar.gz
cat runs/report_bundles/v8_top1_readiness_20260617.tar.gz.sha256
