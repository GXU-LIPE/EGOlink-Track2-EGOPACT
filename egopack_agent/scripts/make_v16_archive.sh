#!/usr/bin/env bash
set -euo pipefail
cd /home/data-gxu/acm/egolink2026-main/code/track2/codex
OUT="reports/V16_GT100_TRAJECTORY_DISTILLATION_VAL41_$(date +%Y%m%d_%H%M%S)_archive.tgz"
tar -czf "$OUT" \
  reports/V16_GT100_POOL_AUDIT_v16_gt100_pool_20260619_2325.md \
  reports/V16_GPT55_RULE_DISTILLATION_skipped_after_transport_timeout.md \
  reports/V16_PROCESS_POLICY_IMPLEMENTATION_20260619_234618.md \
  reports/V16_PROCESS_POLICY_IMPLEMENTATION_20260619_235009.md \
  reports/V16_VAL41_RESULT_v16_gt100_val41_20260619_2352.md \
  reports/V16_FAILURE_DIFF_V14_TO_V16_v16_candidate_selection_20260620_0011.md \
  reports/V16_NEXT_DECISION_20260620_001025.md \
  gt_distill_v16/gt100_pool_manifest.json \
  gt_distill_v16/gpt55_rule_distillation_summary.json \
  gt_distill_v16/tool_sequence_automata.json \
  gt_distill_v16/slot_resolver_rules.json \
  gt_distill_v16/anti_failure_rules.json \
  gt_distill_v16/process_repair_templates.json \
  state/latest_v16_summary.json \
  runs/V16_gt100_distilled_policy_val41/v16_gt100_val41_20260619_2352/partial_eval_summary.json \
  runs/V16_candidate_selection_val41/v16_candidate_selection_20260620_0011/eval_summary.json
echo "$OUT"
