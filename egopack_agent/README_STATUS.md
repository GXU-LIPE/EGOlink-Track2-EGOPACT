# Track2 Codex Status

- updated_at: 2026-06-16T22:47:36+0800
- ego_root: /home/data-gxu/acm/egolink2026-main/code/track2/EgoBench
- codex_root: /home/data-gxu/acm/egolink2026-main/code/track2/codex
- current_strategy: V6 GPT-5.5 visual service agent, commercial API allowed
- official_source_modified: no
- DeepSeek status: preserved as dev baseline, no longer mainline
- GPT-5.5 adapter: implemented
- visual frames/contact sheet: implemented under visual_cache
- visual_state cache: implemented, calls GPT-5.5 at most once per task/video when key exists
- canonical resolver: implemented from official DB init data
- duplicate DB mutation guard: enabled and smoke-tested
- user_id / restaurant_name pinning: enabled
- order product_name -> dish_name top-level alias: enabled
- Annie category alias Steaks -> Selected Steaks: enabled
- final auto-submit: no

## Current Gate Status

- openai_key_present_in_remote_env: no
- GPT-5.5 gate: not run if key_present is no
- latest gate report: reports/02_gpt55_gate_summary_gpt55_v6_gate_20260616_224636.md
- adapter smoke: reports/01_gpt55_adapter_smoke_20260616_224647.md
- implementation report: reports/V6_IMPLEMENTATION_REPORT_20260616_224623.md
- GitHub issue fix audit: reports/00_github_issue_fix_audit_20260616_223134.md

## How To Run GPT-5.5 Gate

Set the key manually in the remote shell/session; do not write it to files:

```bash
export OPENAI_API_KEY=...
cd /home/data-gxu/acm/egolink2026-main/code/track2/codex
bash scripts/run_gpt55_v6_gate.sh
```

The gate runs:

- V6_gpt55_direct
- V6_1_gpt55_guarded
- V6_2_gpt55_visual_state
- V6_3_gpt55_visual_retry

Fixed 4-task gate:

- kitchen2_easy
- order1_easy
- restaurant4_easy
- retail9_easy

## Reports

- reports/00_github_issue_fix_audit_20260616_223134.md
- reports/01_gpt55_adapter_smoke_20260616_224647.md
- reports/02_gpt55_gate_summary_gpt55_v6_gate_20260616_224636.md
- reports/V6_IMPLEMENTATION_REPORT_*.md
- reports/FINAL_MODEL_USAGE_DRAFT_20260616_224736.md
- reports/gpt55_heartbeat_20260616_224736.md
- analysis/gpt55_gate_matrix_*.csv after real gate runs
- state/best_track2_api_version.json

## GPT-5.5 bridge status (2026-06-17T00:47:58+0800)

- Current status: blocked by OpenAI authentication, not by remote network.
- Remote health check to local bridge: OK at `http://127.0.0.1:17901/health`.
- API call result: OpenAI returned `401 invalid_api_key`; key value was not logged.
- Valid GPT-5.5 Track2 gate metrics: not available yet.
- Cleanup: duplicate bridge gate runs were stopped.
- Latest blocker report: `reports/gpt55_bridge_blocker_20260617_004602.md`.
- Rerun after replacing key: `bash scripts/run_gpt55_bridge_gate.sh`.
- Final submission: not submitted.
## GPT-5.5 endpoint status (2026-06-17T11:11:21+0800)

- Endpoint tested: `https://ai-pixel.online/v1` works with `gpt-5.5`; `https://cf.ai-pixel.online` returns 403/1010.
- Key stored in `state/.openai_env` with mode 600; key value not logged.
- Current best API version: `V6_1_3_gpt55_guarded_endpoint`.
- 4-task gate best: joint 50%, result 75%, tool 50%, micro 70.83%.
- Successful joint scenarios: retail9, restaurant4.
- order1: result success now 100%, tool trajectory still mismatch.
- kitchen2: still failing; tool calls reduced from 62 to 35 in V6.1.1, but branch/result still wrong.
- Latest report: `reports/GPT55_ENDPOINT_GATE_SUMMARY_20260617_111121.md`.
- Final submission: not submitted.

## 2026-06-17 GPT-5.5 Targeted Update

- Best remains V6_1_3_gpt55_guarded_endpoint / gpt55_endpoint_gate_20260617_105936.
- Best 4-task metrics: joint 50%, result 75%, tool 50%, micro 70.83%.
- Restored best_track2_api_version.json after an invalid 0-score run had overwritten it.
- Fixed run collector so single-version gates no longer summarize unrelated versions as zeros.
- Main endpoint route is openai_compatible_chat against ai-pixel; adapter route supports base_url but is not selected for main service due soft-failure/low-tool behavior.
- Order diagnostics: /home/data-gxu/acm/egolink2026-main/code/track2/codex/reports/ORDER_PROCESS_ALIGNER_20260617_133208.md
- Kitchen diagnostics: /home/data-gxu/acm/egolink2026-main/code/track2/codex/reports/KITCHEN_BRANCH_REPAIR_20260617_133208.md
- Visual ablation: /home/data-gxu/acm/egolink2026-main/code/track2/codex/reports/VISUAL_PRIOR_ABLATION_20260617_133041.md
- Recommendation: keep V6_1_3 as candidate; next work should add deterministic kitchen branch helper and a more constrained order process-state helper before any larger run.
- Final submission: not submitted.

## 2026-06-17 V7 Human-Prior Agent

- V7 modules installed under `wrappers/egobench_agent_plus/`; official EgoBench source was not modified.
- Tested run: `V7_1_process_verifier` / `human_prior_v71c_20260617_1506`.
- V7_1 4-task metrics: joint 50%, result 50%, tool 50%, micro 62.5%, avg tool calls 19.5.
- Current best remains `V6_1_3_gpt55_guarded_endpoint` / `gpt55_endpoint_gate_20260617_105936` with joint 0.5, result 0.75, tool 0.5, micro 0.7083333333333333.
- V7 preserved retail9 and restaurant4 joint success, but did not improve order1/kitchen2 enough to replace best.
- New traces: `train_data/human_prior_policy_traces.jsonl` (44 records).
- Reports: `HUMAN_PRIOR_IMPLEMENTATION_20260617_155816.md`, `HUMAN_PRIOR_GATE_SUMMARY_20260617_155816.md`, `HUMAN_PRIOR_ABLATION_20260617_155816.md`, `HUMAN_PRIOR_PAPER_NOTES_20260617_155816.md`.
- Final submission: not submitted.

## Official Final Minimal Sync 20260617_170840

- Report: `/home/data-gxu/acm/egolink2026-main/code/track2/codex/reports/OFFICIAL_FINAL_MIN_SYNC_20260617_170840.md`
- Final guide: `/home/data-gxu/acm/egolink2026-main/code/track2/codex/reports/FINAL_STAGE_SUBMISSION_GUIDE_20260617_170840.md`
- Top1 analysis: `/home/data-gxu/acm/egolink2026-main/code/track2/codex/reports/TOP1_READINESS_ANALYSIS_20260617_170840.md`
- Applied: README.md, run_all_scenarios.sh, analysis_scripts/evaluate_interaction.py, scenarios/final/retail6.json, scenarios/final/retail10.json, scenarios/final/kitchen4.json, scenarios/final/restaurant5.json, scenarios/final/order2.json
- No final submission was made.

## Final Compliance Patch 20260617_171726

- Report: /home/data-gxu/acm/egolink2026-main/code/track2/codex/reports/FINAL_COMPLIANCE_PATCH_20260617_171726.md
- Service agent final mode no longer receives final JSON image_description/analysis metadata.
- Packer uses official team_name_track2.zip structure.
- No final submission was made.

## Final Runner Entrypoint Patch 20260617_172012

- Report: `/home/data-gxu/acm/egolink2026-main/code/track2/codex/reports/FINAL_RUNNER_ENTRYPOINT_PATCH_20260617_172012.md`
- `runners/run_all_scenarios_plus.sh --final_eval` now runs official final scenarios with `TRACK2_FINAL_EVAL=1`.
- No final submission was made.

## V8 Dataset Count Audit 20260617_180809

- Report: `/home/data-gxu/acm/egolink2026-main/code/track2/codex/reports/V8_DATASET_COUNT_AUDIT_20260617_180809.md`
- Final total: 309/309, final GT present: False
- Dev/offline measurable tasks: 736
- Split state: `/home/data-gxu/acm/egolink2026-main/code/track2/codex/state/track2_data_split_20260617_180809.json`
- No final submission was made.

## V8 Implementation 20260617_181223

- Report: `/home/data-gxu/acm/egolink2026-main/code/track2/codex/reports/V8_IMPLEMENTATION_20260617_181223.md`
- Added switch-controlled order/kitchen helpers, V8 event logger, DeepSeek crosscheck scaffold, and multicandidate reranker scaffold.
- Current best remains V6_1_3 unless validation_A/B both improve.
- No final submission was made.

## V8_SMOKE_SUMMARY v8_smoke_20260617_1823

- Report: `/home/data-gxu/acm/egolink2026-main/code/track2/codex/reports/V8_SMOKE_SUMMARY_v8_smoke_20260617_1823.md`
- version: `V8_6_top1_candidate`
- joint: 0.2500, result: 0.5000, tool: 0.2500, micro: 0.0000
- Best not updated automatically. No final submission was made.

## V8_SMOKE_SUMMARY v8_0_smoke_20260617_1832

- Report: `/home/data-gxu/acm/egolink2026-main/code/track2/codex/reports/V8_SMOKE_SUMMARY_v8_0_smoke_20260617_1832.md`
- version: `V8_0_v6_stable_reval`
- joint: 0.5000, result: 0.5000, tool: 0.5000, micro: 0.0000
- Best not updated automatically. No final submission was made.

## V8_SMOKE_SUMMARY v8_smoke_20260617_1823 Corrected

- Report: `/home/data-gxu/acm/egolink2026-main/code/track2/codex/reports/V8_SMOKE_SUMMARY_v8_smoke_20260617_1823_corrected.md`
- version: `V8_6_top1_candidate`
- joint: 0.2500, result: 0.5000, tool: 0.2500, micro: 0.4545, avg_task_accuracy: 0.5000
- Recomputed from existing result files only; no model/API calls. Best not updated. No final submission was made.

## V8_SMOKE_SUMMARY v8_0_smoke_20260617_1832 Corrected

- Report: `/home/data-gxu/acm/egolink2026-main/code/track2/codex/reports/V8_SMOKE_SUMMARY_v8_0_smoke_20260617_1832_corrected.md`
- version: `V8_0_v6_stable_reval`
- joint: 0.5000, result: 0.5000, tool: 0.5000, micro: 0.6364, avg_task_accuracy: 0.7083
- Recomputed from existing result files only; no model/API calls. Best not updated. No final submission was made.

## V8_SMOKE_SUMMARY v8_1_smoke_20260617_continue

- Report: `/home/data-gxu/acm/egolink2026-main/code/track2/codex/reports/V8_SMOKE_SUMMARY_v8_1_smoke_20260617_continue.md`
- version: `V8_1_order_helper`
- joint: 0.5000, result: 0.5000, tool: 0.5000, micro: 0.5455, avg_task_accuracy: 0.6250
- Tool call match counts: 6/11 gt, interaction_calls=87
- Best not updated automatically. No final submission was made.

## V8_SMOKE_SUMMARY v8_2_smoke_20260617_continue

- Report: `/home/data-gxu/acm/egolink2026-main/code/track2/codex/reports/V8_SMOKE_SUMMARY_v8_2_smoke_20260617_continue.md`
- version: `V8_2_kitchen_helper`
- joint: 0.7500, result: 0.7500, tool: 0.7500, micro: 0.8182, avg_task_accuracy: 0.8750
- Tool call match counts: 9/11 gt, interaction_calls=127
- Best not updated automatically. No final submission was made.

## V8_VALIDATION_A_SMALL v8_2_valA_small_20260617_continue

- Report: `/home/data-gxu/acm/egolink2026-main/code/track2/codex/reports/V8_VALIDATION_A_SMALL_v8_2_valA_small_20260617_continue.md`
- version: `V8_2_kitchen_helper`
- joint: 0.0500, result: 0.1500, tool: 0.0500, micro: 0.2471, avg_task_accuracy: 0.2354
- Tool call match counts: 21/85 gt, interaction_calls=394
- Best not updated automatically. No final submission was made.

## V8_VALIDATION_A_SMALL v8_0_valA_small_20260617_continue

- Report: `/home/data-gxu/acm/egolink2026-main/code/track2/codex/reports/V8_VALIDATION_A_SMALL_v8_0_valA_small_20260617_continue.md`
- version: `V8_0_v6_stable_reval`
- joint: 0.0500, result: 0.1000, tool: 0.0500, micro: 0.1059, avg_task_accuracy: 0.1125
- Tool call match counts: 9/85 gt, interaction_calls=324
- Best not updated automatically. No final submission was made.

## V8 Top1 Readiness 20260617_210513

- Report: `/home/data-gxu/acm/egolink2026-main/code/track2/codex/reports/V8_TOP1_READINESS_20260617_210513.md`
- Decision: do not update protected best; no final submission.
- Best remains: `V6_1_3_gpt55_guarded_endpoint` / `gpt55_endpoint_gate_20260617_105936`.
- Fixed V8 validation micro extraction and recomputed smoke metrics.
- V8_2 smoke improved to joint 0.75 / micro 0.8182, but validation_A_small was only joint 0.05 / result 0.15 / tool 0.05 / micro 0.2471.
- Next: mine validation_A failures and improve order/retail/restaurant process coverage before validation_B or final.

## V8_VALIDATION_A_SMALL v613_valA_small_20260617_quick

- Report: `/home/data-gxu/acm/egolink2026-main/code/track2/codex/reports/V8_VALIDATION_A_SMALL_v613_valA_small_20260617_quick.md`
- version: `V6_1_3_gpt55_guarded_endpoint`
- joint: 0.0500, result: 0.1000, tool: 0.0500, micro: 0.3059, avg_task_accuracy: 0.1934
- Tool call match counts: 26/85 gt, interaction_calls=535
- Best not updated automatically. No final submission was made.

## V8_VALIDATION_A_SMALL V9_2_scoring_prompt_soft_guard_validation_A_small_20260618_010142

- Report: `/home/data-gxu/acm/egolink2026-main/code/track2/codex/reports/V8_VALIDATION_A_SMALL_V9_2_scoring_prompt_soft_guard_validation_A_small_20260618_010142.md`
- version: `V9_2_scoring_prompt_soft_guard`
- joint: 0.0500, result: 0.0500, tool: 0.0500, micro: 0.2706, avg_task_accuracy: 0.1434
- Tool call match counts: 23/85 gt, interaction_calls=445
- Best not updated automatically. No final submission was made.

## V8_VALIDATION_A_SMALL V9_4_memory_retrieval_validation_A_small_20260618_014932

- Report: `/home/data-gxu/acm/egolink2026-main/code/track2/codex/reports/V8_VALIDATION_A_SMALL_V9_4_memory_retrieval_validation_A_small_20260618_014932.md`
- version: `V9_4_memory_retrieval`
- joint: 0.1000, result: 0.1500, tool: 0.1000, micro: 0.3529, avg_task_accuracy: 0.2400
- Tool call match counts: 30/85 gt, interaction_calls=428
- Best not updated automatically. No final submission was made.

## V9_VALIDATION_A_SMALL_COMPARISON 20260618_022913

- Report: `/home/data-gxu/acm/egolink2026-main/code/track2/codex/reports/V9_VALIDATION_A_SMALL_COMPARISON_20260618_022913.md`
- V9_4_memory_retrieval: joint 0.1000, result 0.1500, tool 0.1000, micro 0.3529, avg_task_accuracy 0.2400.
- Compared with V6_1_3 validation_A_small: joint 0.0500, result 0.1000, tool 0.0500, micro 0.3059.
- Candidate recorded in `state/v9_candidate_version.json`; protected best not updated.
- Next: run validation_A_medium for V9_4 before validation_B/final.

## V8_VALIDATION_A_MEDIUM V9_4_memory_retrieval_validation_A_medium_20260618_023033 Corrected

- Report: `/home/data-gxu/acm/egolink2026-main/code/track2/codex/reports/V8_VALIDATION_A_MEDIUM_V9_4_memory_retrieval_validation_A_medium_20260618_023033_corrected.md`
- version: `V9_4_memory_retrieval`
- joint: 0.0488, result: 0.0976, tool: 0.0488, micro: 0.1795, avg_task_accuracy: 0.1793
- Recomputed from existing result files only; no model/API calls. Best not updated. No final submission was made.

## V9 Validation A Medium Stop 20260618_034143

- Report: `/home/data-gxu/acm/egolink2026-main/code/track2/codex/reports/V9_VALIDATION_A_MEDIUM_20260618_034143.md`
- Readiness: `/home/data-gxu/acm/egolink2026-main/code/track2/codex/reports/V9_TOP1_READINESS_20260618_034143.md`
- V9_4 A_medium recomputed: valid 41/42, joint 0.0488, result 0.0976, tool 0.0488, micro 0.1795.
- Decision: stopped before validation_B_holdout; protected best unchanged; no final submission.

## V8_VALIDATION_A_MEDIUM V9_4_5_visual_retrieval_fix_validation_A_medium_20260618_1014

- Report: `/home/data-gxu/acm/egolink2026-main/code/track2/codex/reports/V8_VALIDATION_A_MEDIUM_V9_4_5_visual_retrieval_fix_validation_A_medium_20260618_1014.md`
- version: `V9_4_5_visual_retrieval_fix`
- joint: 0.1220, result: 0.1707, tool: 0.1220, micro: 0.2500, avg_task_accuracy: 0.2489
- Tool call match counts: 39/156 gt, interaction_calls=809
- Best not updated automatically. No final submission was made.

## V8_VALIDATION_A_MEDIUM V9_5_memory_deepseek_rerank_validation_A_medium_20260618_1218_noproxy

- Report: `/home/data-gxu/acm/egolink2026-main/code/track2/codex/reports/V8_VALIDATION_A_MEDIUM_V9_5_memory_deepseek_rerank_validation_A_medium_20260618_1218_noproxy.md`
- version: `V9_5_memory_deepseek_rerank`
- joint: 0.0500, result: 0.1750, tool: 0.0750, micro: 0.2583, avg_task_accuracy: 0.2510
- Tool call match counts: 39/151 gt, interaction_calls=979
- Best not updated automatically. No final submission was made.

## V9_5 A_medium Rerank 20260618_140514

- Report: `/home/data-gxu/acm/egolink2026-main/code/track2/codex/reports/V9_5_DEEPSEEK_RERANKER_20260618_140514.md`
- V9_5 A_medium: joint 5.00%, result 17.50%, tool 7.50%, micro 25.83%, valid 40.
- V9_4_5 A_medium remains stronger on joint: 12.20%; protected best unchanged.
- Retail trim events: 9; no completed-run API transport soft failures after no-proxy launcher.
- No validation_B_holdout, no final, no protected best update.

## V8_VALIDATION_A_MEDIUM V9_5_memory_deepseek_rerank_validation_A_medium_20260618_1420_frozen_noproxy

- Report: `/home/data-gxu/acm/egolink2026-main/code/track2/codex/reports/V8_VALIDATION_A_MEDIUM_V9_5_memory_deepseek_rerank_validation_A_medium_20260618_1420_frozen_noproxy.md`
- version: `V9_5_memory_deepseek_rerank`
- joint: 0.1220, result: 0.1707, tool: 0.1220, micro: 0.2628, avg_task_accuracy: 0.2482
- Tool call match counts: 41/156 gt, interaction_calls=974
- Best not updated automatically. No final submission was made.

## V9.5 Frozen A Medium 20260618_155051

- Report: `/home/data-gxu/acm/egolink2026-main/code/track2/codex/reports/V9_5_FROZEN_A_MEDIUM_ANALYSIS_20260618_155051.md`
- run_id: `V9_5_memory_deepseek_rerank_validation_A_medium_20260618_1420_frozen_noproxy`
- valid: 41, joint: 0.1220, result: 0.1707, tool: 0.1220, micro: 0.2628
- Split hygiene fixed with materialized validation_A_limit30; protected best unchanged; no final submission.

## V8_VALIDATION_A_MEDIUM V10_full_memory_final_candidate_draft_A_medium_sanity_20260618_1716

- Report: `/home/data-gxu/acm/egolink2026-main/code/track2/codex/reports/V8_VALIDATION_A_MEDIUM_V10_full_memory_final_candidate_draft_A_medium_sanity_20260618_1716.md`
- version: `V10_full_memory_final_candidate_draft`
- joint: 0.1220, result: 0.1707, tool: 0.1220, micro: 0.2949, avg_task_accuracy: 0.2834
- Tool call match counts: 46/156 gt, interaction_calls=936
- Best not updated automatically. No final submission was made.

## V10 Final Candidate Draft Package 20260619_001219

- final_full_sanity: `/home/data-gxu/acm/egolink2026-main/code/track2/codex/reports/V10_FINAL_FULL_SANITY_20260619_001219.md`
- submission_zip: `/home/data-gxu/acm/egolink2026-main/code/track2/codex/submissions/V10_full_memory_final_candidate_draft_track2.zip`
- auto_submitted: no

## V8_VALIDATION_A_MEDIUM V12_official_style_qwen3vl_memory_validation_A_medium_20260619_1226

- Report: `/home/data-gxu/acm/egolink2026-main/code/track2/codex/reports/V8_VALIDATION_A_MEDIUM_V12_official_style_qwen3vl_memory_validation_A_medium_20260619_1226.md`
- version: `V12_official_style_qwen3vl_memory`
- joint: 0.0732, result: 0.1707, tool: 0.0976, micro: 0.2564, avg_task_accuracy: 0.2570
- Tool call match counts: 40/156 gt, interaction_calls=1086
- Best not updated automatically. No final submission was made.

## V8_VALIDATION_A_MEDIUM V12_official_style_qwen3vl_memory_val41_all_qwen3vl_20260619_143308

- Report: `/home/data-gxu/acm/egolink2026-main/code/track2/codex/reports/V8_VALIDATION_A_MEDIUM_V12_official_style_qwen3vl_memory_val41_all_qwen3vl_20260619_143308.md`
- version: `V12_official_style_qwen3vl_memory`
- joint: 0.0000, result: 0.0000, tool: 0.0000, micro: 0.0000, avg_task_accuracy: 0.0000
- Tool call match counts: 0/0 gt, interaction_calls=0
- Best not updated automatically. No final submission was made.
