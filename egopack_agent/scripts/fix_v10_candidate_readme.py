#!/usr/bin/env python3
from pathlib import Path

path = Path("/home/data-gxu/acm/egolink2026-main/code/track2/codex/reports/V10_FINAL_CANDIDATE_README_20260619_001219.md")
path.write_text(
    """# V10 Final Candidate README

- memory bank dev/offline task count: see `/home/data-gxu/acm/egolink2026-main/code/track2/codex/reports/V10_FULL_MEMORY_BANK_BUILD_20260618_171530.md`; 649 tasks entered memory in the completed build.
- V10 A_medium vs V9_5: joint equal at 12.20%; micro improved from 26.28% to 29.49%.
- final smoke: passed.
- final 309 completed: True
- submission zip: `/home/data-gxu/acm/egolink2026-main/code/track2/codex/submissions/V10_full_memory_final_candidate_draft_track2.zip`
- compliance risk detected by sanity checks: False
- recommendation: manual review before any submission; this script did not submit anything.

- final sanity: `/home/data-gxu/acm/egolink2026-main/code/track2/codex/reports/V10_FINAL_FULL_SANITY_20260619_001219.md`
- package report: `/home/data-gxu/acm/egolink2026-main/code/track2/codex/reports/V10_SUBMISSION_PACKAGE_DRAFT_20260619_001219.md`
- technical report pdf: `/home/data-gxu/acm/egolink2026-main/code/track2/codex/reports/V10_full_memory_final_candidate_draft.pdf`

## Known Quality Risks

- Three final tasks produced no tool calls (`order2_easy` task 18, `restaurant5_easy` tasks 7 and 14). These are visual grounding failures around pointed dishes/drinks where the agent replied instead of retrieving.
- The `api_error_hits` count in the sanity report is conservative string matching; a recursive audit found no non-empty `api_error` fields. See `reports/V10_RESULT_RISK_AUDIT_20260619_0012.jsonl`.
- This remains a complete candidate package, but manual review is recommended before submission.
""",
    encoding="utf-8",
)
print(path)
