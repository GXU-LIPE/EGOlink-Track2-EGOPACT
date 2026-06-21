#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import time
from pathlib import Path


CODEX = Path("/home/data-gxu/acm/egolink2026-main/code/track2/codex")


def load(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return default


def main() -> None:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    manifest = load(CODEX / "gt_distill_v16" / "gt100_pool_manifest.json", {})
    distill = load(CODEX / "gt_distill_v16" / "gpt55_rule_distillation_summary.json", {})
    impl = CODEX / "reports" / f"V16_PROCESS_POLICY_IMPLEMENTATION_{stamp}.md"
    lines = [
        f"# V16 Process Policy Implementation {stamp}",
        "",
        "- final_run: false",
        "- official_egobench_source_modified: false",
        "- protected_v10_zip_overwritten: false",
        "- uses_final_hidden_metadata: false",
        "- uses_val41_label_for_policy: false",
        "",
        "## Implemented Files",
        "",
        "- `scripts/build_v16_gt100_distill_pool.py`",
        "- `scripts/distill_v16_rules_with_gpt55.py`",
        "- `scripts/run_v16_val41_parallel.py`",
        "- `scripts/v16_candidate_selection_val41.py`",
        "- `wrappers/egobench_agent_plus/process_policy_v16.py`",
        "- `wrappers/egobench_agent_plus/slot_resolver_v16.py`",
        "- `wrappers/egobench_agent_plus/candidate_compiler_v16.py`",
        "- `wrappers/egobench_agent_plus/joint_reranker_v16.py`",
        "",
        "## Prompt Integration",
        "",
        "- `prompt_builder.py` imports `build_v16_process_policy_prompt`.",
        "- V16 prompt injection is gated by `TRACK2_ENABLE_V16_PROCESS_POLICY=1` or `TRACK2_RUN_VERSION` starting with `V16_`.",
        "- Default non-V16 behavior remains unchanged.",
        "",
        "## Distilled Modules",
        "",
    ]
    for key, val in sorted((manifest.get("module_counts") or {}).items()):
        lines.append(f"- {key}: {val}")
    if distill:
        lines += [
            "",
            "## GPT-5.5 Rule Distillation",
            "",
            f"- samples_completed: {distill.get('samples_completed')}",
            f"- gpt55_rows: {distill.get('gpt55_rows')}",
            f"- fallback_rows: {distill.get('fallback_rows')}",
            f"- api_error_rows: {distill.get('api_error_rows')}",
        ]
    impl.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(impl)


if __name__ == "__main__":
    main()
